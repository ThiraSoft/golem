package vk

// The logit head on the card when the file keeps it in four bits.
//
// vk/q6k.go is the same tensor in the format Gemma's QAT builds publish it in.
// A Qwen3 checkpoint quantized with `llama-quantize --pure` keeps its head in
// Q4_0 like everything else, and a Q4_0 product against a Q8_0 activation is
// already written: shaders/matvec.comp, the kernel the shared branch's down
// projection and the attention's four projections use. What is here is the
// tensor made resident and the activation handed over — no shader of its own.

import (
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// A Q40Head is one Q4_0 matrix resident in device memory, with the small
// per-token buffers around it.
type Q40Head struct {
	d          *Device
	rows, cols int

	weights *Buffer // device local, uploaded once
	aq      *Buffer // the activation, Q8_0
	as      *Buffer // its scales, then its corrections
	out     *Buffer // one float per row

	pipe   *Pipeline
	set    *Set
	groups uint32
}

// NewQ40Head uploads a Q4_0 matrix. data is the tensor exactly as the file
// holds it: rows of cols/32 blocks of eighteen bytes.
func NewQ40Head(d *Device, data []byte, rows, cols int) (*Q40Head, error) {
	if cols%nn.QuantBlock != 0 {
		return nil, fmt.Errorf("vk: a Q4_0 row needs a multiple of %d columns, given %d", nn.QuantBlock, cols)
	}
	if want := rows * rowBytesQ4_0(cols); len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns need %d bytes, given %d", rows, cols, want, len(data))
	}
	h := &Q40Head{d: d, rows: rows, cols: cols}

	var err error
	if h.weights, err = d.Upload(splitQ4_0(data, rows, cols)); err != nil {
		return nil, err
	}
	nb := cols / nn.QuantBlock
	for _, spec := range []struct {
		into **Buffer
		size int
		back bool
	}{
		{&h.aq, cols, false},
		{&h.as, 2 * nb * 4, false},
		{&h.out, rows * 4, true},
	} {
		var b *Buffer
		if spec.back {
			b, err = d.Readback(spec.size, bufferUsageStorage)
		} else {
			b, err = d.Host(spec.size, bufferUsageStorage)
		}
		if err != nil {
			h.Close()
			return nil, err
		}
		*spec.into = b
	}

	buffers := []*Buffer{h.weights, h.aq, h.as, h.out}
	if h.pipe, err = d.NewPipeline(matvecSPIRV, len(buffers), uint32(unsafe.Sizeof(moePush{}))); err != nil {
		h.Close()
		return nil, err
	}
	if h.set, err = h.pipe.NewSet(buffers); err != nil {
		h.Close()
		return nil, err
	}
	// A workgroup writes matvecOuts rows. Vulkan promises 65535 workgroups on
	// an axis, and a vocabulary of a quarter of a million rows is a tenth of
	// that, so this is a count rather than a ceiling — but say so, because a
	// larger vocabulary would fail silently.
	groups := (rows + matvecOuts - 1) / matvecOuts
	if groups > 65535 {
		h.Close()
		return nil, fmt.Errorf("vk: %d rows want %d workgroups, past the 65535 an axis holds", rows, groups)
	}
	h.groups = uint32(groups)
	return h, nil
}

// MatVec computes y = W*x for a batch of one that already carries its Q8_0
// form, and leaves the result in out.
//
// A batch of one, because nn.Batch interleaves its quantized form by block and
// then by column — the layout the CPU kernels read a row of every column with
// — and a column of that is contiguous only when there is one. The same
// restriction is on Mixture.Run, for the same reason.
func (h *Q40Head) MatVec(b *nn.Batch, out []float32) error {
	if b.Q == nil {
		return fmt.Errorf("vk: a Q4_0 product needs the activation in its Q8_0 form")
	}
	if b.Size != 1 {
		return fmt.Errorf("vk: the head reads one column at a time, given a batch of %d", b.Size)
	}
	if b.Width != h.cols {
		return fmt.Errorf("vk: the head reads %d inputs, the activation has %d", h.cols, b.Width)
	}
	if len(out) != h.rows {
		return fmt.Errorf("vk: the head writes %d outputs, given %d", h.rows, len(out))
	}

	nb := h.cols / nn.QuantBlock
	dst := h.aq.Bytes()
	for i, v := range b.Q[:h.cols] {
		dst[i] = byte(v)
	}
	scales := h.as.Floats()
	copy(scales[:nb], b.Scales[:nb])
	copy(scales[nb:2*nb], b.Corr[:nb])

	push := moePush{dim: uint32(h.rows), ffn: uint32(h.cols), used: 1}
	if err := h.set.Dispatch(h.groups, unsafe.Pointer(&push)); err != nil {
		return err
	}
	copy(out, h.out.Floats())
	return nil
}

func (h *Q40Head) Close() {
	if h.set != nil {
		h.set.Close()
		h.set = nil
	}
	if h.pipe != nil {
		h.pipe.Close()
		h.pipe = nil
	}
	for _, b := range []**Buffer{&h.out, &h.as, &h.aq, &h.weights} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}
