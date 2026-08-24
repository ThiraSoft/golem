package vk

// The logit head on the card.
//
// Gemma ties its head to its input embedding, so the largest tensor in the
// file is also a matrix of a quarter of a million rows that has to be read in
// full for every token drawn. On this machine that is 577 mebibytes at
// something near the memory bus's ceiling — a quarter of what a token costs,
// and a quarter that no amount of CPU work can shorten, because the bytes are
// the cost. Moving it to a card whose memory is an order of magnitude wider is
// the one thing that changes the arithmetic.

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/q6k.comp -o shaders/q6k.spv

//go:embed shaders/q6k.spv
var q6kSPIRV []byte

const (
	// q6kSourceBytes is a superblock as the file stores it.
	q6kSourceBytes = 210
	// q6kPaddedBytes is a superblock as the shader reads it. The two extra
	// bytes buy every load its alignment; shaders/q6k.comp says why.
	q6kPaddedBytes = 212
	// q6kTeamsPerGroup is how many rows one workgroup carries, and must match
	// the TEAMS constant in the shader.
	q6kTeamsPerGroup = 4
)

// A Q6KHead is one Q6_K matrix resident in device memory, with the small
// per-token buffers around it.
type Q6KHead struct {
	d           *Device
	rows, cols  int
	superblocks int

	weights *Buffer // device local, uploaded once
	act     *Buffer // the Q8_K magnitudes
	scales  *Buffer // one float per superblock
	sums    *Buffer // sixteen group sums per superblock
	out     *Buffer // one float per row

	pipe   *Pipeline
	set    *Set
	groups uint32
}

// push is the shader's push constant block.
type q6kPush struct {
	rows        uint32
	superblocks uint32
}

// NewQ6KHead uploads a Q6_K matrix. data is the tensor exactly as the file
// holds it: rows of cols/256 superblocks of 210 bytes.
func NewQ6KHead(d *Device, data []byte, rows, cols int) (*Q6KHead, error) {
	if cols%nn.SuperBlock != 0 {
		return nil, fmt.Errorf("vk: a Q6_K row needs a multiple of %d columns, given %d", nn.SuperBlock, cols)
	}
	sb := cols / nn.SuperBlock
	if want := rows * sb * q6kSourceBytes; len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns need %d bytes, given %d", rows, cols, want, len(data))
	}

	h := &Q6KHead{d: d, rows: rows, cols: cols, superblocks: sb}

	// The repack. It runs once, and it is the only copy of the weights that
	// ever exists on this side: the padded form is built a chunk at a time
	// straight into the staging path.
	padded := make([]byte, rows*sb*q6kPaddedBytes)
	for i := 0; i < rows*sb; i++ {
		copy(padded[i*q6kPaddedBytes:], data[i*q6kSourceBytes:(i+1)*q6kSourceBytes])
	}
	var err error
	if h.weights, err = d.Upload(padded); err != nil {
		return nil, err
	}
	padded = nil

	if h.act, err = d.Host(cols, bufferUsageStorage); err != nil {
		h.Close()
		return nil, err
	}
	if h.scales, err = d.Host(sb*4, bufferUsageStorage); err != nil {
		h.Close()
		return nil, err
	}
	if h.sums, err = d.Host(sb*16*4, bufferUsageStorage); err != nil {
		h.Close()
		return nil, err
	}
	if h.out, err = d.Readback(rows*4, bufferUsageStorage); err != nil {
		h.Close()
		return nil, err
	}

	buffers := []*Buffer{h.weights, h.act, h.scales, h.sums, h.out}
	if h.pipe, err = d.NewPipeline(q6kSPIRV, len(buffers), uint32(unsafe.Sizeof(q6kPush{}))); err != nil {
		h.Close()
		return nil, err
	}
	if h.set, err = h.pipe.NewSet(buffers); err != nil {
		h.Close()
		return nil, err
	}
	// Each workgroup carries q6kTeamsPerGroup rows. Vulkan only promises 65535
	// workgroups on an axis, and the shader loops over rows with the grid as
	// its stride, so this is a ceiling and not a shape.
	h.groups = uint32(min((rows+q6kTeamsPerGroup-1)/q6kTeamsPerGroup, 65535))
	return h, nil
}

// MatVec computes y = W*x for one column of a batch that already carries its
// Q8_K form, and leaves the result in out.
func (h *Q6KHead) MatVec(b *nn.Batch, column int, out []float32) error {
	if b.QK == nil {
		return fmt.Errorf("vk: a Q6_K product needs the activation in its Q8_K form")
	}
	if b.Width != h.cols {
		return fmt.Errorf("vk: the head reads %d inputs, the activation has %d", h.cols, b.Width)
	}
	if len(out) != h.rows {
		return fmt.Errorf("vk: the head writes %d outputs, given %d", h.rows, len(out))
	}

	q := b.QK[column*b.Width : (column+1)*b.Width]
	dst := h.act.Bytes()
	for i, v := range q {
		dst[i] = byte(v)
	}
	copy(h.scales.Floats(), b.KScales[column*h.superblocks:(column+1)*h.superblocks])
	sums := unsafe.Slice((*int32)(unsafe.Pointer(&h.sums.Bytes()[0])), h.superblocks*16)
	src := b.BSums[column*h.superblocks*16 : (column+1)*h.superblocks*16]
	for i, v := range src {
		sums[i] = int32(v)
	}

	push := q6kPush{rows: uint32(h.rows), superblocks: uint32(h.superblocks)}
	if err := h.set.Dispatch(h.groups, unsafe.Pointer(&push)); err != nil {
		return err
	}
	copy(out, h.out.Floats())
	return nil
}

func (h *Q6KHead) Close() {
	if h.set != nil {
		h.set.Close()
		h.set = nil
	}
	if h.pipe != nil {
		h.pipe.Close()
		h.pipe = nil
	}
	for _, b := range []**Buffer{&h.out, &h.sums, &h.scales, &h.act, &h.weights} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}
