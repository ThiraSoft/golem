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
// What crosses the bus is both what a sequence reconstructs and the path it
// walked. The research bench measures a reconstruction and wants only the
// first; a file stores the path, and stores it rather than the reconstruction
// because the decoder recomputes one from the other. It cannot be the
// processor's path either: TestViterbiMatchesCPU holds the two encoders to the
// same cost and not to the same path, so the bytes written have to be the ones
// the card actually walked.
//
// The path comes back wide, one state a weight, and is packed into the file's
// 520-bit sequences by nn.PutT4GStates — the same division of labour
// vk/encode_golem.go makes, and for the same reason: a shader that packed
// twelve-bit fields straddling words would cost more to write than the packing
// costs to do on the processor.

import (
	_ "embed"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// spread runs fn over [0,n) split across the processors. The card's passes are
// bracketed by host-side traffic — staging the floats in, taking the
// reconstruction and the path back out — and that traffic is a whole pass over
// sixteen million weights each way, through memory the driver maps rather than
// the allocator. Left on one thread it was the conversion: one core pegged and
// the card at a fifth of its occupancy, waiting for its next pass to be handed
// to it.
func spread(n int, fn func(lo, hi int)) {
	w := runtime.GOMAXPROCS(0)
	if w > n {
		w = n
	}
	if w < 2 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	chunk := (n + w - 1) / w
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); fn(lo, hi) }(lo, hi)
	}
	wg.Wait()
}

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/viterbi_tcq.comp -o shaders/viterbi_tcq.spv

//go:embed shaders/viterbi_tcq.spv
var viterbiTCQSPIRV []byte

//go:generate glslc -O -DKBITS=5 --target-env=vulkan1.1 -fshader-stage=compute shaders/viterbi_tcq.comp -o shaders/viterbi_tcq5.spv

// viterbiTCQ5SPIRV is the wide tier's, for the logit head. Widening k narrows
// the trellis rather than the planes — the prefix count is 2^(L−k) — so the
// backpointers halve in number and double in width, and the whole thing still
// fits the sixty-four kibibytes.
//
//go:embed shaders/viterbi_tcq5.spv
var viterbiTCQ5SPIRV []byte

//go:generate glslc -O -DKBITS=3 --target-env=vulkan1.1 -fshader-stage=compute shaders/viterbi_tcq.comp -o shaders/viterbi_tcq3.spv

// viterbiTCQ3SPIRV is the narrow tier's. Narrowing k widens the trellis rather
// than the planes — the prefix count is 2^(L−k) — so the backpointers double in
// number and halve in width, which is the wide tier's trade run backwards.
//
//go:embed shaders/viterbi_tcq3.spv
var viterbiTCQ3SPIRV []byte

// What the shader is compiled for. They are not parameters: the sequence
// length is what makes the backpointers fit beside the two cost planes in a
// workgroup's sixty-four kibibytes, and changing either means recompiling with
// the arithmetic redone. shaders/viterbi_tcq.comp carries that arithmetic.
const (
	TrellisGPUSeq = 128 // weights coded as one sequence
	TrellisGPUK   = 4   // bits emitted per weight in the ordinary tier
	TrellisGPUK5  = 5   // and in the wide one, which is the logit head's
	TrellisGPUK3  = 3   // and in the narrow one
	TrellisGPUL   = 12  // state bits
)

// TrellisGPUHasK says whether a kernel was built for that rate.
func TrellisGPUHasK(k int) bool {
	return k == TrellisGPUK || k == TrellisGPUK5 || k == TrellisGPUK3
}

// maxWorkgroups is what every Vulkan implementation promises on the first axis
// of a dispatch. Cards allow more and this one allows a great deal more, but a
// pass that stays inside the promise costs nothing to arrange.
const maxWorkgroups = 65535

type viterbiPush struct {
	seqs uint32
	gain float32
}

// TrellisEncoder holds the room for one pass of weights.
type TrellisEncoder struct {
	d   *Device
	cap int
	k   int

	stage    *Buffer // host, what a pass is written into
	z        *Buffer // device, what the kernel reads
	rec      *Buffer // device, the reconstruction it writes
	recBack  *Buffer // host, what comes back
	path     *Buffer // device, the state of each weight
	pathBack *Buffer // host, the same

	pipe map[int]*Pipeline
	set  map[int]*Set
}

// NewTrellisEncoder builds the encoder. capacity is how many weights one pass
// carries; the buffers are sixteen bytes a weight, so sixteen million of them
// is a quarter of a gigabyte and any matrix goes through in passes of that.
func NewTrellisEncoder(d *Device, capacity int) (*TrellisEncoder, error) {
	if capacity%TrellisGPUSeq != 0 {
		return nil, fmt.Errorf("vk: a pass of %d weights is not a whole number of %d-weight sequences",
			capacity, TrellisGPUSeq)
	}
	e := &TrellisEncoder{d: d, cap: capacity, k: TrellisGPUK,
		pipe: map[int]*Pipeline{}, set: map[int]*Set{}}
	for _, b := range []struct {
		at   **Buffer
		host bool
		use  uint32
	}{
		{&e.stage, true, bufferUsageTransferSrc},
		{&e.z, false, bufferUsageStorage | bufferUsageTransferDst},
		{&e.rec, false, bufferUsageStorage | bufferUsageTransferSrc},
		{&e.recBack, true, bufferUsageTransferDst},
		{&e.path, false, bufferUsageStorage | bufferUsageTransferSrc},
		{&e.pathBack, true, bufferUsageTransferDst},
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
	bufs := []*Buffer{e.z, e.rec, e.path}
	for k, spirv := range map[int][]byte{
		TrellisGPUK: viterbiTCQSPIRV, TrellisGPUK5: viterbiTCQ5SPIRV, TrellisGPUK3: viterbiTCQ3SPIRV,
	} {
		pipe, err := d.NewPipeline(spirv, len(bufs), uint32(unsafe.Sizeof(viterbiPush{})))
		if err != nil {
			e.Close()
			return nil, err
		}
		e.pipe[k] = pipe
		set, err := pipe.NewSet(bufs)
		if err != nil {
			e.Close()
			return nil, err
		}
		e.set[k] = set
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
	return e.quantize(norm, gain, nil)
}

// UseK picks the tier the next passes run at. It is a mode rather than an
// argument because a converter sets it once per tensor and the buffers are the
// same either way.
func (e *TrellisEncoder) UseK(k int) error {
	if !TrellisGPUHasK(k) {
		return fmt.Errorf("vk: there is no Viterbi kernel at %d bits a weight", k)
	}
	e.k = k
	return nil
}

// QuantizePath is Quantize with the path kept: states holds one state a weight,
// which is what nn.PutT4GStates writes into a file. The reconstruction is still
// written back over norm, because the converter needs it to fit each block's
// step by least squares before it packs anything.
func (e *TrellisEncoder) QuantizePath(norm []float32, gain float32, states []uint16) error {
	if len(states) != len(norm) {
		return fmt.Errorf("vk: %d weights and room for %d states", len(norm), len(states))
	}
	return e.quantize(norm, gain, states)
}

func (e *TrellisEncoder) quantize(norm []float32, gain float32, states []uint16) error {
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
		spread(n, func(lo, hi int) { copy(xs[lo:hi], norm[at+lo:at+hi]) })
		seqs := n / TrellisGPUSeq
		push := viterbiPush{seqs: uint32(seqs), gain: gain}
		// One submission: the weights across, the kernel, the answer back. A
		// fence a pass rather than a fence a sequence.
		err := e.d.Submit(func(r *Recorder) {
			r.CopyFrom(e.z, 0, e.stage, 0, n*4)
			r.Barrier()
			r.Dispatch(e.set[e.k], uint32(seqs), unsafe.Pointer(&push))
			r.Barrier()
			r.CopyFrom(e.recBack, 0, e.rec, 0, n*4)
			r.CopyFrom(e.pathBack, 0, e.path, 0, n*4)
		})
		if err != nil {
			return err
		}
		rec := e.recBack.Floats()[:n]
		path := e.pathBack.Uints()[:n]
		spread(n, func(lo, hi int) {
			copy(norm[at+lo:at+hi], rec[lo:hi])
			if states == nil {
				return
			}
			// The path comes back a word a weight and is stored a half-word:
			// the state is twelve bits. Narrowing it is one instruction and
			// sixteen million of them, and it reads from a buffer the driver
			// maps, so it is bus-bound rather than arithmetic-bound and every
			// thread is worth having.
			for i := lo; i < hi; i++ {
				states[at+i] = uint16(path[i])
			}
		})
	}
	return nil
}

func (e *TrellisEncoder) Close() {
	if e == nil {
		return
	}
	for _, b := range []*Buffer{e.stage, e.z, e.rec, e.recBack, e.path, e.pathBack} {
		if b != nil {
			b.Close()
		}
	}
	for _, s := range e.set {
		s.Close()
	}
	e.set = nil
	for _, p := range e.pipe {
		p.Close()
	}
	e.pipe = nil
}
