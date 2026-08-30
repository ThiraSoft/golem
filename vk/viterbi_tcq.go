package vk

// The trellis encoder on the card.
//
// The lattice's encoder searches a step over some forty candidates, each eight
// roundings: a few thousand operations a scale block. The trellis's is a
// Viterbi over 2^L states, which is four thousand operations *a weight* at
// L=12 — five hundred times the work, and it was twenty-six minutes for a
// six-hundred-million-weight model on twelve cores, which makes three hours
// for a four-billion one. That is the number this file exists to remove.
//
// What crosses the bus is what a sequence reconstructs, not its codes. The
// research path measures a reconstruction, so this returns one; packing a path
// into a file's bit stream is a shift and a mask per weight and belongs with
// the format, not here.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/viterbi_tcq.comp -o shaders/viterbi_tcq.spv

//go:embed shaders/viterbi_tcq.spv
var viterbiTCQSPIRV []byte

// What the shader is compiled for. They are not parameters: the sequence
// length is what makes the backpointers fit beside the two cost planes in a
// workgroup's sixty-four kibibytes, and changing either means recompiling with
// the arithmetic redone. shaders/viterbi_tcq.comp carries that arithmetic.
const (
	TrellisGPUSeq = 128 // weights coded as one sequence
	TrellisGPUK   = 4   // bits emitted per weight
	TrellisGPUL   = 12  // state bits
)

type viterbiPush struct {
	seqs uint32
	gain float32
}

// TrellisEncoder holds the room for one pass of weights.
type TrellisEncoder struct {
	d   *Device
	cap int

	stage   *Buffer // host, what a pass is written into
	z       *Buffer // device, what the kernel reads
	rec     *Buffer // device, what it writes
	recBack *Buffer // host, what comes back

	pipe *Pipeline
	set  *Set
}

// NewTrellisEncoder builds the encoder. capacity is how many weights one pass
// carries; the buffers are sixteen bytes a weight, so sixteen million of them
// is a quarter of a gigabyte and any matrix goes through in passes of that.
func NewTrellisEncoder(d *Device, capacity int) (*TrellisEncoder, error) {
	if capacity%TrellisGPUSeq != 0 {
		return nil, fmt.Errorf("vk: a pass of %d weights is not a whole number of %d-weight sequences",
			capacity, TrellisGPUSeq)
	}
	e := &TrellisEncoder{d: d, cap: capacity}
	for _, b := range []struct {
		at   **Buffer
		host bool
		use  uint32
	}{
		{&e.stage, true, bufferUsageTransferSrc},
		{&e.z, false, bufferUsageStorage | bufferUsageTransferDst},
		{&e.rec, false, bufferUsageStorage | bufferUsageTransferSrc},
		{&e.recBack, true, bufferUsageTransferDst},
	} {
		var err error
		if b.host {
			*b.at, err = d.Host(capacity*4, b.use)
		} else {
			*b.at, err = d.Local(capacity*4, b.use)
		}
		if err != nil {
			e.Close()
			return nil, err
		}
	}
	bufs := []*Buffer{e.z, e.rec}
	var err error
	if e.pipe, err = d.NewPipeline(viterbiTCQSPIRV, len(bufs), uint32(unsafe.Sizeof(viterbiPush{}))); err != nil {
		e.Close()
		return nil, err
	}
	if e.set, err = e.pipe.NewSet(bufs); err != nil {
		e.Close()
		return nil, err
	}
	return e, nil
}

// Capacity is how many weights one pass carries.
func (e *TrellisEncoder) Capacity() int { return e.cap }

// Quantize runs the whole array through the trellis, one sequence of
// TrellisGPUSeq weights at a time, and writes the reconstruction back over it.
// norm is expected already normalised — unit RMS per scale block — which is
// what compress.VQ hands its quantizer.
func (e *TrellisEncoder) Quantize(norm []float32, gain float32) error {
	if len(norm)%TrellisGPUSeq != 0 {
		return fmt.Errorf("vk: %d weights is not a whole number of %d-weight sequences",
			len(norm), TrellisGPUSeq)
	}
	if gain == 0 {
		gain = 1
	}
	// A workgroup a sequence, and a dispatch may have only so many of them.
	// Vulkan promises 65535 on the first axis; this card allows far more, but a
	// pass sized past the promise would be silently wrong on a driver that
	// keeps to it — and silently wrong is the one thing a converter cannot be.
	step := e.cap
	if lim := maxWorkgroups * TrellisGPUSeq; step > lim {
		step = lim
	}
	xs := e.stage.Floats()
	for at := 0; at < len(norm); at += step {
		n := step
		if at+n > len(norm) {
			n = len(norm) - at
		}
		copy(xs[:n], norm[at:at+n])
		seqs := n / TrellisGPUSeq
		push := viterbiPush{seqs: uint32(seqs), gain: gain}
		// One submission: the weights across, the kernel, the answer back. A
		// fence a pass rather than a fence a sequence.
		err := e.d.Submit(func(r *Recorder) {
			r.CopyFrom(e.z, 0, e.stage, 0, n*4)
			r.Barrier()
			r.Dispatch(e.set, uint32(seqs), unsafe.Pointer(&push))
			r.Barrier()
			r.CopyFrom(e.recBack, 0, e.rec, 0, n*4)
		})
		if err != nil {
			return err
		}
		copy(norm[at:at+n], e.recBack.Floats()[:n])
	}
	return nil
}

func (e *TrellisEncoder) Close() {
	if e == nil {
		return
	}
	for _, b := range []*Buffer{e.stage, e.z, e.rec, e.recBack} {
		if b != nil {
			b.Close()
		}
	}
	if e.set != nil {
		e.set.Close()
	}
	if e.pipe != nil {
		e.pipe.Close()
	}
}
