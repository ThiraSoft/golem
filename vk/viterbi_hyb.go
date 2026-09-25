package vk

// The pair trellis's encoder on the card: nn/pair.go's formats, and
// shaders/viterbi_hyb.comp's arithmetic. Its host side is TrellisEncoder's —
// stage the weights, one submission a pass, the reconstruction and the path
// back — with a codebook bound beside them, because an H3G state's value is a
// table entry and not a hash.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O -DLBITS=14 -DKV=6 --target-env=vulkan1.1 -fshader-stage=compute shaders/viterbi_hyb.comp -o shaders/viterbi_hyb.spv
//go:generate glslc -O -DLBITS=15 -DKV=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/viterbi_hyb.comp -o shaders/viterbi_hyb4.spv

//go:embed shaders/viterbi_hyb.spv
var viterbiHybSPIRV []byte

//go:embed shaders/viterbi_hyb4.spv
var viterbiHyb4SPIRV []byte

// PairGPUSeq is the sequence shaders/viterbi_hyb.comp is compiled for.
const PairGPUSeq = 128 // weights a sequence, two a state

// PairGPUHas says whether a kernel was built for a state of l bits at k bits a
// weight: H3G's fourteen at three, H4G's fifteen at four.
func PairGPUHas(l, k int) bool { return pairSPIRV(l, k) != nil }

func pairSPIRV(l, k int) []byte {
	switch {
	case l == 14 && k == 3:
		return viterbiHybSPIRV
	case l == 15 && k == 4:
		return viterbiHyb4SPIRV
	}
	return nil
}

type pairPush struct {
	seqs   uint32
	eshift uint32
	emask  uint32
}

// PairEncoder holds the room for one pass of weights.
type PairEncoder struct {
	d   *Device
	cap int

	stage    *Buffer
	z        *Buffer
	rec      *Buffer
	recBack  *Buffer
	path     *Buffer
	pathBack *Buffer
	book     *Buffer

	pipe   *Pipeline
	set    *Set
	eshift uint32
	emask  uint32
}

// NewPairEncoder builds the encoder for a trellis of l state bits at k bits a
// weight, whose states read entry bits shift.. of s·(s+1) out of book, a pair
// an entry and a power of two of them.
func NewPairEncoder(d *Device, capacity int, book []float32, l, k, shift int) (*PairEncoder, error) {
	spirv := pairSPIRV(l, k)
	if spirv == nil {
		return nil, fmt.Errorf("vk: there is no pair Viterbi for %d state bits at %d bits a weight", l, k)
	}
	if n := len(book) / 2; n&(n-1) != 0 {
		return nil, fmt.Errorf("vk: a codebook of %d pairs is not a power of two", n)
	}
	if capacity%PairGPUSeq != 0 {
		return nil, fmt.Errorf("vk: a pass of %d weights is not a whole number of %d-weight sequences",
			capacity, PairGPUSeq)
	}
	e := &PairEncoder{d: d, cap: capacity, eshift: uint32(shift), emask: uint32(len(book)/2 - 1)}
	for _, b := range []struct {
		at   **Buffer
		host bool
		size int
		use  uint32
	}{
		{&e.stage, true, capacity * 4, bufferUsageTransferSrc},
		{&e.z, false, capacity * 4, bufferUsageStorage | bufferUsageTransferDst},
		{&e.rec, false, capacity * 4, bufferUsageStorage | bufferUsageTransferSrc},
		{&e.recBack, true, capacity * 4, bufferUsageTransferDst},
		{&e.path, false, capacity * 2, bufferUsageStorage | bufferUsageTransferSrc},
		{&e.pathBack, true, capacity * 2, bufferUsageTransferDst},
	} {
		var err error
		if b.host {
			*b.at, err = d.Host(b.size, b.use)
		} else {
			*b.at, err = d.Local(b.size, b.use)
		}
		if err != nil {
			e.Close()
			return nil, err
		}
	}
	var err error
	if e.book, err = d.Upload(floatBytes(book)); err != nil {
		e.Close()
		return nil, err
	}
	bufs := []*Buffer{e.z, e.rec, e.path, e.book}
	if e.pipe, err = d.NewPipeline(spirv, len(bufs), uint32(unsafe.Sizeof(pairPush{}))); err != nil {
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
func (e *PairEncoder) Capacity() int { return e.cap }

// QuantizePath runs norm through the pair trellis, writes the reconstruction
// back over it, and fills states with one state a pair.
func (e *PairEncoder) QuantizePath(norm []float32, states []uint16) error {
	if len(norm)%PairGPUSeq != 0 {
		return fmt.Errorf("vk: %d weights is not a whole number of %d-weight sequences", len(norm), PairGPUSeq)
	}
	if len(states) != len(norm)/2 {
		return fmt.Errorf("vk: %d weights and room for %d states", len(norm), len(states))
	}
	step := e.cap
	if lim := maxWorkgroups * PairGPUSeq; step > lim {
		step = lim
	}
	xs := e.stage.Floats()
	for at := 0; at < len(norm); at += step {
		n := min(step, len(norm)-at)
		spread(n, func(lo, hi int) { copy(xs[lo:hi], norm[at+lo:at+hi]) })
		seqs := n / PairGPUSeq
		push := pairPush{seqs: uint32(seqs), eshift: e.eshift, emask: e.emask}
		err := e.d.Submit(func(r *Recorder) {
			r.CopyFrom(e.z, 0, e.stage, 0, n*4)
			r.Barrier()
			r.Dispatch(e.set, uint32(seqs), unsafe.Pointer(&push))
			r.Barrier()
			r.CopyFrom(e.recBack, 0, e.rec, 0, n*4)
			r.CopyFrom(e.pathBack, 0, e.path, 0, n*2)
		})
		if err != nil {
			return err
		}
		rec := e.recBack.Floats()[:n]
		path := e.pathBack.Uints()[:n/2]
		spread(n, func(lo, hi int) { copy(norm[at+lo:at+hi], rec[lo:hi]) })
		spread(n/2, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				states[at/2+i] = uint16(path[i])
			}
		})
	}
	return nil
}

func (e *PairEncoder) Close() {
	if e == nil {
		return
	}
	for _, b := range []*Buffer{e.stage, e.z, e.rec, e.recBack, e.path, e.pathBack, e.book} {
		if b != nil {
			b.Close()
		}
	}
	if e.set != nil {
		e.set.Close()
		e.set = nil
	}
	if e.pipe != nil {
		e.pipe.Close()
		e.pipe = nil
	}
}
