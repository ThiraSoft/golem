package vk

// The logit head on the card, whatever format the file wrote it in.
//
// vk/q6k.go moved a Q6_K head off the processor and vk/q40head.go a Q4_0 one,
// each with its own upload, its own pipeline and its own dispatch. A third
// format arrived — a Q8_0 checkpoint of the 26B A4B, whose tied embedding is
// 784 megabytes — and the shape of the answer was already written down twice.
// So this is the third one written once: the format door of
// vk/quantproduct.go decides the layout and the kernel, and everything else
// here is the same four buffers those two files each had.
//
// What it is worth is not subtle. On that checkpoint the head was the whole of
// what a token waited on — 92.02 ms of 152.90, sixty per cent — because a
// quarter of a million rows of eight-bit weights crossed a PCIe 3.0 x8 link
// for every token drawn. It is the largest tensor in the model and it is read
// in full each time; there is no arithmetic that shortens it, only a shorter
// wire.

import (
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// A Head is the head matrix on the card. Two implement it: this one, over any
// format the door reads, and vk.Q6KHead, which keeps a kernel of its own
// because a Q6_K head against a Q8_K activation is measurably better than the
// same matrix against a Q8_0 one.
type Head interface {
	// Prepare puts the activation in the quantized form this head reads. A
	// caller does not otherwise know which that is.
	Prepare(b *nn.Batch)
	MatVec(b *nn.Batch, column int, out []float32) error
	// Capped says whether the head caps its own logits, so that a caller does
	// not cap them twice.
	Capped() bool
	// Table is the matrix itself, so a caller holding both this and a Stack
	// can read a row of it there: a tied head and its input embedding are the
	// same tensor, and uploading it twice would be its whole size for nothing.
	Table() (*Buffer, int)
	Close()
}

var _ Head = (*QuantHead)(nil)
var _ Head = (*Q6KHead)(nil)

// A QuantHead is one quantized matrix resident in device memory, with the
// small per-token buffers around it.
type QuantHead struct {
	d          *Device
	q          nn.Quant
	rows, cols int

	weights *Buffer // device local, uploaded once
	aq      *Buffer // the activation's eight-bit magnitudes
	as      *Buffer // its scales, then its corrections
	out     *Buffer // one float a row

	pipe   *Pipeline
	set    *Set
	groups uint32
}

// NewQuantHead uploads a head matrix. data is the tensor exactly as the file
// holds it, and q the format the file declared — never one inferred from the
// length, because the lengths collide: eighteen bytes to a block of thirty-two
// is Q4_0 and it is also Q4_K.
func NewQuantHead(d *Device, q nn.Quant, data []byte, rows, cols int) (*QuantHead, error) {
	if !QuantReadable(q) {
		return nil, fmt.Errorf("vk: there is no head kernel for %s", q)
	}
	layout, err := quantLayout(q, data, rows, cols)
	if err != nil {
		return nil, err
	}
	h := &QuantHead{d: d, q: q, rows: rows, cols: cols}
	fail := func(err error) (*QuantHead, error) {
		h.Close()
		return nil, err
	}
	if h.weights, err = d.Upload(layout); err != nil {
		return nil, err
	}
	layout = nil

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
			return fail(err)
		}
		*spec.into = b
	}

	// The door's pipeline, at the widths a token takes. coop is false: a head
	// answers one column, and the tiled forms a prompt uses are never reached
	// from here.
	if h.pipe, err = newQuantProduct(d, q, false); err != nil {
		return fail(err)
	}
	if h.set, err = h.pipe.NewSet([]*Buffer{h.weights, h.aq, h.as, h.out}); err != nil {
		return fail(err)
	}
	// A workgroup writes quantOuts(q) rows, which is not the same number for
	// every format — the Q8_0 kernels are built at the shader's own default
	// and the rest at a measured shape. Vulkan promises 65535 workgroups on an
	// axis, and a vocabulary of a quarter of a million rows is well inside it,
	// so this is a count rather than a ceiling — but say so, because a larger
	// vocabulary would otherwise answer short and say nothing.
	outs := quantOuts(q)
	groups := (rows + outs - 1) / outs
	if groups > 65535 {
		return fail(fmt.Errorf("vk: %d rows want %d workgroups, past the 65535 an axis holds", rows, groups))
	}
	h.groups = uint32(groups)
	return h, nil
}

// Prepare puts the activation in the eight-bit form every kernel behind the
// door reads.
func (h *QuantHead) Prepare(b *nn.Batch) { b.Quantize() }

// MatVec computes y = W*x for a batch of one that already carries its Q8_0
// form, and leaves the result in out.
//
// A batch of one, because nn.Batch interleaves its quantized form by block and
// then by column — the layout the CPU kernels read a row of every column with
// — and a column of that is contiguous only when there is one. The same
// restriction is on Mixture.Run, for the same reason.
func (h *QuantHead) MatVec(b *nn.Batch, column int, out []float32) error {
	if b.Q == nil {
		return fmt.Errorf("vk: a %s product needs the activation in its Q8_0 form", h.q)
	}
	if b.Size != 1 || column != 0 {
		return fmt.Errorf("vk: the head reads one column at a time, given column %d of a batch of %d", column, b.Size)
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

// Capped is false: the door's kernels write a raw logit, and a model with a
// final softcap applies it on the answer. That is a quarter of a million
// hyperbolic tangents on the processor, and it is left there deliberately —
// see gemma's Logits, which measures it against what the head saved.
func (h *QuantHead) Capped() bool { return false }

// Table is the matrix itself, in the layout its kernels read.
func (h *QuantHead) Table() (*Buffer, int) { return h.weights, h.cols }

// Quant is the format the matrix is stored in.
func (h *QuantHead) Quant() nn.Quant { return h.q }

func (h *QuantHead) Close() {
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
