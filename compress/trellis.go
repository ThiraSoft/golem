package compress

// Trellis-coded quantization: the codebook with no dimension.
//
// A lattice quantizes d weights at a time and pays for its shell with a table,
// which is why this repository settled on D4 — E8 at the same rate wants two
// hundred mebibytes of decode table and a workgroup has thirty-two kibibytes.
// The trellis escapes that trade entirely. It codes a whole sequence, T weights
// long, as one path through a state machine, so the effective dimension is T
// rather than four; and the value of a state is *computed* from the state
// rather than looked up, so there is no table at all.
//
// The structure is QTIP's bitshift trellis (Tseng et al., NeurIPS 2024). The
// state is simply the last L bits of the code stream, so weight t reads the L
// bits at offset t·k and hashes them. Nothing about the trellis has to be
// stored, and every weight decodes independently of its neighbours — which is
// what a shader needs, and what a conventional trellis cannot give.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// TrellisCode names how a state becomes a number.
type TrellisCode int

const (
	// Code1MAD is QTIP's lookup-free code: a linear congruential step, then
	// the sum of the four bytes of the result. Four bytes summed is already
	// close enough to a Gaussian for a source the rotation has made Gaussian,
	// and the whole thing is a multiply, an add and a byte-sum.
	Code1MAD TrellisCode = iota
	// Code3INST is the other lookup-free code: the LCG's output is split into
	// two half-precision numbers by xor against a magic constant and added.
	Code3INST
)

const (
	mad1A = 34038481
	mad1B = 76625530
	i3A   = 89226354
	i3B   = 64248484
	i3M   = 0.922

	// 1/147.8, written as the float32 it is rather than as a division.
	//
	// A division is not a multiply by the reciprocal: twenty of 1MAD's 1021
	// values differ by one unit in the last place between the two forms. A
	// shader compiler is free to pick either, and a Viterbi is a chain of
	// comparisons over a codebook with only 1021 distinct values for 4096
	// states — near-ties are everywhere, and one bit of difference in a value
	// moves whole paths. It cost seven percent of the weights, at identical
	// error, before both sides were made to multiply by this.
	mad1Scale = 0.00676589971
)

// trellisValue turns an L-bit state into the number it reconstructs.
func trellisValue(c TrellisCode, s uint32) float32 {
	switch c {
	case Code3INST:
		x := uint32(i3A)*s + uint32(i3B)
		m1 := math.Float32frombits(((x >> 16) ^ 0x3c00) & 0xffff << 16)
		m2 := math.Float32frombits((x^0x3c00)&0xffff<<16) * float32(i3M)
		return m1 + m2
	default:
		// nn owns this, because the decoder does: a file is read by nn and a
		// second copy of the hash here would be a second format sharing a
		// name. At L past sixteen the state no longer fits what nn takes, and
		// nothing in the format goes there.
		if l := s >> 16; l == 0 {
			return nn.T4GValue(uint16(s))
		}
		x := uint32(mad1A)*s + uint32(mad1B)
		sum := (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff)
		return (float32(sum) - 510) * mad1Scale
	}
}

// TrellisTable precomputes the value of every state. At L=16 it is 256 KiB and
// it is built once for the whole model; the shader does not need it, because
// the code recomputes what this table remembers.
func TrellisTable(c TrellisCode, l int) []float32 {
	t := make([]float32, 1<<uint(l))
	for s := range t {
		t[s] = trellisValue(c, uint32(s))
	}
	return t
}

// TrellisOpts is what a trellis costs and how much memory its path is allowed.
type TrellisOpts struct {
	K    int // bits emitted per weight; the rate, before the scales
	L    int // state bits: how far back the path remembers
	Seq  int // weights coded as one sequence — the effective dimension
	Code TrellisCode

	// Gain narrows the codebook against the source. The block arrives at unit
	// RMS and the code's values are N(0,1), so a gain of one lines the two up
	// — which is not what minimises the error. The reconstruction that does is
	// shrunk towards zero, so the source is scaled up before rounding and the
	// path scaled back down after. It is the trellis's answer to the step
	// search the lattice does per block, except that the rotation makes every
	// block the same Gaussian, so one number serves the whole model and costs
	// no bits at all. Zero means one.
	Gain float64
}

func (t TrellisOpts) gain() float32 {
	if t.Gain == 0 {
		return 1
	}
	return float32(t.Gain)
}

// BitsPerSeq is what one sequence actually occupies. The first weight needs a
// whole L-bit window before there is any history to shift, so a sequence is not
// T·k bits but T·k + (L−k) — under two percent at T=256, and the honest number
// to put in a rate.
func (t TrellisOpts) BitsPerSeq() int { return (t.Seq-1)*t.K + t.L }

func (t TrellisOpts) BPW() float64 {
	return float64(t.BitsPerSeq()) / float64(t.Seq)
}

// viterbi finds the cheapest path through the bitshift trellis for one
// sequence and writes the reconstruction back over z.
//
// The saving that makes this affordable: a state s can only have been reached
// from a state whose low L−k bits are s>>k, so all 2^k predecessors of s share
// one prefix, and every state sharing that prefix shares the same minimum. One
// step is therefore 2^L work rather than 2^L·2^k, which is what turns the
// algorithm from hopeless into an afternoon.
type viterbiWork struct {
	prev, cur []float32
	bp        []uint8 // per step, the winning j for each prefix
	nstates   int
	nprefix   int
}

func newViterbiWork(o TrellisOpts) *viterbiWork {
	ns := 1 << uint(o.L)
	np := 1 << uint(o.L-o.K)
	return &viterbiWork{
		prev:    make([]float32, ns),
		cur:     make([]float32, ns),
		bp:      make([]uint8, o.Seq*np),
		nstates: ns,
		nprefix: np,
	}
}

// viterbi finds the path and writes the reconstruction over z. states, when it
// is not nil, is filled with the state each weight was coded as — which is what
// a file stores, the reconstruction being what the decoder recomputes from it.
func viterbi(z []float32, val []float32, o TrellisOpts, w *viterbiWork, states []uint16) {
	ns, np := w.nstates, w.nprefix
	kb := uint(o.K)
	shift := uint(o.L - o.K)
	T := len(z)

	// t = 0: any window is a legal start, so the cost is the distortion alone.
	for s := 0; s < ns; s++ {
		d := z[0] - val[s]
		w.prev[s] = d * d
	}

	for t := 1; t < T; t++ {
		bp := w.bp[t*np : (t+1)*np]
		// For each prefix p, the best predecessor among the 2^k states that
		// end in it. Every state s with s>>k == p inherits this minimum.
		for p := 0; p < np; p++ {
			best, bj := float32(math.MaxFloat32), 0
			for j := 0; j < 1<<kb; j++ {
				if c := w.prev[(j<<shift)|p]; c < best {
					best, bj = c, j
				}
			}
			bp[p] = uint8(bj)
			// Every s whose prefix is p: s = (p << k) | b for b in [0, 2^k).
			base := p << kb
			for b := 0; b < 1<<kb; b++ {
				s := base | b
				d := z[t] - val[s]
				w.cur[s] = best + d*d
			}
		}
		w.prev, w.cur = w.cur, w.prev
	}

	// Traceback from the cheapest final state.
	end, bestCost := 0, float32(math.MaxFloat32)
	for s := 0; s < ns; s++ {
		if w.prev[s] < bestCost {
			bestCost, end = w.prev[s], s
		}
	}
	s := end
	for t := T - 1; t >= 0; t-- {
		z[t] = val[s]
		if states != nil {
			states[t] = uint16(s)
		}
		if t > 0 {
			j := int(w.bp[t*np+(s>>kb)])
			s = (j << shift) | (s >> kb)
		}
	}
}

// TrellisAccel, when set, is given first refusal on a matrix. It is how the
// card gets to do the Viterbi without this package importing the one that
// drives it — vk imports compress, so the arrow cannot point both ways. It
// returns false for a shape it was not compiled for, and the processor takes
// it from there.
var TrellisAccel func(norm []float32, o TrellisOpts) bool

// TrellisPathAccel is TrellisAccel for a caller that wants the path as well as
// the reconstruction — the converter, which has a file to write. It is a second
// hook rather than a wider first one because the research bench asks for a
// reconstruction and has nowhere to put a path, and a pass that carried one
// anyway would be four bytes a weight across the bus for nothing.
var TrellisPathAccel func(norm []float32, o TrellisOpts, states []uint16) bool

// quantizeTrellis runs the whole normalised matrix through the trellis, one
// sequence at a time, writing the reconstruction back over norm.
func quantizeTrellis(norm []float32, o TrellisOpts, val []float32, states []uint16) {
	if states == nil {
		if TrellisAccel != nil && TrellisAccel(norm, o) {
			return
		}
	} else if TrellisPathAccel != nil && TrellisPathAccel(norm, o, states) {
		return
	}
	nseq := len(norm) / o.Seq
	g := o.gain()
	Parallel(nseq, func(lo, hi int) {
		w := newViterbiWork(o)
		z := make([]float32, o.Seq)
		for i := lo; i < hi; i++ {
			seq := norm[i*o.Seq : (i+1)*o.Seq]
			for j, v := range seq {
				z[j] = v * g
			}
			var st []uint16
			if states != nil {
				st = states[i*o.Seq : (i+1)*o.Seq]
			}
			viterbi(z, val, o, w, st)
			for j, v := range z {
				seq[j] = v / g
			}
		}
	})
}

// QuantizeTrellis is quantizeTrellis for callers outside this package — the
// card's encoder, which has to be held to exactly what the processor does.
func QuantizeTrellis(norm []float32, o TrellisOpts) {
	quantizeTrellis(norm, o, TrellisTable(o.Code, o.L), nil)
}

// QuantizeTrellisPath is QuantizeTrellis with the path kept: states holds the
// state of every weight, one per weight, which is what a file is written from.
//
// The reconstruction is still written back over norm, and the converter needs
// it: each block's step is fitted to it by least squares before anything is
// packed, and the step is chosen after the path for the reason
// compress/codec.go gives.
func QuantizeTrellisPath(norm []float32, o TrellisOpts, states []uint16) {
	quantizeTrellis(norm, o, TrellisTable(o.Code, o.L), states)
}
