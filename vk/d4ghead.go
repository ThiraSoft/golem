package vk

// The logit head on the card when the file is a .golem.
//
// The head is the embedding read the other way round, and in this format it is
// a site like any other: its columns were scaled and rotated when it was
// written, so the hidden state meets the reciprocal of that scale and the same
// rotation before the product. That is the one thing here that vk/q40head.go
// does not have, and it is two dispatches rather than one.
//
// It is also the largest tensor in the model — a quarter of a million rows on
// a Qwen3 vocabulary — so it is the read that decides what a token costs once
// the blocks are on the card.

import (
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// A D4GHead is the head matrix resident in device memory, with the transform
// its activation goes through in front of it.
type D4GHead struct {
	d          *Device
	k          *D4GKernels
	rows, cols int

	act  *Buffer // the hidden state, written by the host and transformed in place
	out  *Buffer // one float a row
	prep *PrepareD4G
	m    *D4GMatrix
}

// NewD4GHead uploads the head. data is the tensor as the file holds it, and
// pre is the vector the checkpoint carries for it — output.pre, the same one
// the input path undoes a row at a time.
func NewD4GHead(k *D4GKernels, data []byte, rows, cols int, pre []float32) (*D4GHead, error) {
	if len(pre) != cols {
		return nil, fmt.Errorf("vk: the head's vector is %d wide, the head reads %d", len(pre), cols)
	}
	h := &D4GHead{d: k.d, k: k, rows: rows, cols: cols}
	fail := func(err error) (*D4GHead, error) {
		h.Close()
		return nil, err
	}
	var err error
	if h.act, err = k.d.Host(cols*4, bufferUsageStorage); err != nil {
		return fail(err)
	}
	if h.out, err = k.d.Readback(rows*4, bufferUsageStorage); err != nil {
		return fail(err)
	}
	if h.prep, err = NewPrepareD4G(k.d, h.act, pre, prepareD4GGroup); err != nil {
		return fail(err)
	}
	if h.m, err = NewD4GMatrixOn(k, data, rows, cols, h.act, h.out); err != nil {
		return fail(err)
	}
	// A workgroup writes d4gRowsPerGroup rows. Vulkan promises 65535 on an
	// axis, and a vocabulary of a quarter of a million rows wants sixteen
	// thousand — a count rather than a ceiling, but say so, because a larger
	// vocabulary would fail silently.
	if g := h.m.Groups(); g > 65535 {
		return fail(fmt.Errorf("vk: %d rows want %d workgroups, past the 65535 an axis holds", rows, g))
	}
	return h, nil
}

// Logits computes the whole vocabulary for one hidden state, which arrives as
// the model's final norm left it — unscaled and unrotated, because this does
// both.
func (h *D4GHead) Logits(hidden, out []float32) error {
	if len(hidden) != h.cols {
		return fmt.Errorf("vk: the head reads %d inputs, given %d", h.cols, len(hidden))
	}
	if len(out) != h.rows {
		return fmt.Errorf("vk: the head writes %d outputs, given %d", h.rows, len(out))
	}
	copy(h.act.Floats(), hidden)
	push := h.prep.Push(1)
	if err := h.prep.Set().Dispatch(h.prep.Groups(1), unsafe.Pointer(&push)); err != nil {
		return err
	}
	p := h.m.Push(0)
	if err := h.m.Set(1).Dispatch(h.m.Groups(), unsafe.Pointer(&p)); err != nil {
		return err
	}
	copy(out, h.out.Floats())
	return nil
}

// Table is the head matrix and its step grid, so that a caller holding both
// this and a Stack can have the card look a token up for itself.
func (h *D4GHead) Table() (weights, steps *Buffer, cols int) {
	return h.m.weights, h.k.table, h.cols
}

// Quant is the format the head is stored in, which the stack needs to pick the
// kernel that reads a row of it.
func (h *D4GHead) Quant() nn.Quant { return h.k.q }

func (h *D4GHead) Close() {
	if h.m != nil {
		h.m.Close()
		h.m = nil
	}
	if h.prep != nil {
		h.prep.Close()
		h.prep = nil
	}
	for _, b := range []*Buffer{h.act, h.out} {
		if b != nil {
			b.Close()
		}
	}
	h.act, h.out = nil, nil
}
