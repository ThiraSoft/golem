package compress

// The trellis that codes two weights a state: nn/pair.go's H3G and H4G.
//
// The Viterbi is compress/trellis.go's with one change of unit. A step is a
// pair of weights and not one, the stream moves 2k bits a step, and the
// distortion of a state is the squared distance between two points in the
// plane. The saving that makes it affordable is the same: every predecessor of
// a state s ends in s's top L−2k bits, so one minimum per prefix serves all
// 2^2k states that share it, and a step is 2^L work. At L=14 that is four
// times T3G's per step and two times per weight.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// PairOpts is a pair trellis.
type PairOpts struct {
	K       int // bits a weight; a state adds 2K
	L       int // state bits
	Seq     int // weights a sequence, two a state
	Shift   int // a state's entry is bits Shift.. of s·(s+1)
	Entries int // pairs in the codebook
}

// Entry is which codebook entry a state reads.
func (o PairOpts) Entry(s uint16) int { return nn.PairEntry(s, o.Shift, o.Entries) }

// PairOptsFor is the trellis a pair tier is written with.
func PairOptsFor(q nn.Quant) PairOpts {
	p := nn.PairTierOf(q)
	return PairOpts{K: p.K, L: p.L, Seq: nn.T4GSeq, Shift: p.Shift, Entries: p.Entries}
}

type pairWork struct {
	prev, cur []float32
	bp        []uint8
}

func newPairWork(o PairOpts) *pairWork {
	ns := 1 << uint(o.L)
	np := 1 << uint(o.L-2*o.K)
	return &pairWork{
		prev: make([]float32, ns),
		cur:  make([]float32, ns),
		bp:   make([]uint8, o.Seq/2*np),
	}
}

// viterbiPairs finds the cheapest path for one sequence, writes the
// reconstruction back over z, and returns its cost. val holds a pair a state.
// states, when not nil, receives the state of each pair.
//
// The arithmetic is written out — a difference, its square, a sum, in that
// order and no fused multiply-add — because the card's encoder is held to the
// same cost, and a Viterbi is a chain of comparisons in which one rounding
// moves a near-tie and a near-tie moves a path.
func viterbiPairs(z []float32, val []float32, o PairOpts, w *pairWork, states []uint16) float64 {
	ns := 1 << uint(o.L)
	kv := uint(2 * o.K)
	np := 1 << (uint(o.L) - kv)
	shift := uint(o.L) - kv
	T := len(z) / 2

	dist := func(t, s int) float32 {
		d0 := z[2*t] - val[2*s]
		d1 := z[2*t+1] - val[2*s+1]
		a := d0 * d0
		b := d1 * d1
		return a + b
	}
	for s := 0; s < ns; s++ {
		w.prev[s] = dist(0, s)
	}
	for t := 1; t < T; t++ {
		bp := w.bp[t*np : (t+1)*np]
		for p := 0; p < np; p++ {
			best, bj := float32(math.MaxFloat32), 0
			for j := 0; j < 1<<kv; j++ {
				if c := w.prev[(j<<shift)|p]; c < best {
					best, bj = c, j
				}
			}
			bp[p] = uint8(bj)
			base := p << kv
			for b := 0; b < 1<<kv; b++ {
				s := base | b
				w.cur[s] = best + dist(t, s)
			}
		}
		w.prev, w.cur = w.cur, w.prev
	}
	end, bestCost := 0, float32(math.MaxFloat32)
	for s := 0; s < ns; s++ {
		if w.prev[s] < bestCost {
			bestCost, end = w.prev[s], s
		}
	}
	s := end
	for t := T - 1; t >= 0; t-- {
		z[2*t], z[2*t+1] = val[2*s], val[2*s+1]
		if states != nil {
			states[t] = uint16(s)
		}
		if t > 0 {
			j := int(w.bp[t*np+(s>>kv)])
			s = (j << shift) | (s >> kv)
		}
	}
	return float64(bestCost)
}

// PairAccel, when set, is given first refusal on a whole pass, as
// TrellisPathAccel is for the one-weight trellis. book is the codebook, a pair
// an entry, and a state reads the entry o.Entry names; states receives one
// state a pair. It returns false for a shape it was not built for.
var PairAccel func(norm []float32, o PairOpts, book []float32, states []uint16) bool

// QuantizePairsPath runs the normalised matrix through the pair trellis with
// the codebook book, writes the reconstruction back over norm, and fills
// states with a state a pair — half as many as there are weights.
func QuantizePairsPath(norm []float32, o PairOpts, book []float32, states []uint16) {
	if PairAccel != nil && PairAccel(norm, o, book, states) {
		return
	}
	quantizePairsCPU(norm, o, pairTable(o, book), states)
}

// pairTable is every state's pair under a codebook.
func pairTable(o PairOpts, book []float32) []float32 {
	ns := 1 << uint(o.L)
	val := make([]float32, 2*ns)
	for s := 0; s < ns; s++ {
		e := o.Entry(uint16(s))
		val[2*s], val[2*s+1] = book[2*e], book[2*e+1]
	}
	return val
}

func quantizePairsCPU(norm []float32, o PairOpts, val []float32, states []uint16) float64 {
	nseq := len(norm) / o.Seq
	costs := make([]float64, nseq)
	Parallel(nseq, func(lo, hi int) {
		w := newPairWork(o)
		for i := lo; i < hi; i++ {
			var st []uint16
			if states != nil {
				st = states[i*o.Seq/2 : (i+1)*o.Seq/2]
			}
			costs[i] = viterbiPairs(norm[i*o.Seq:(i+1)*o.Seq], val, o, w, st)
		}
	})
	var total float64
	for _, c := range costs {
		total += c
	}
	return total
}

// QuantizePairsCPU is the processor's encoder alone, for the test that holds
// the card's to it. It returns the summed cost of every path.
func QuantizePairsCPU(norm []float32, o PairOpts, val []float32, states []uint16) float64 {
	return quantizePairsCPU(norm, o, val, states)
}

// TrainPairCodebook is Lloyd's algorithm run through the trellis: code the
// source, then move every codebook entry to the mean of the pairs its states
// coded, and again. book holds the pairs and is updated in place; entry says
// which entry a state reads through o.Entry. It returns the mean squared
// error of each round, measured before that round's update.
func TrainPairCodebook(src []float32, o PairOpts, book []float32, rounds int) []float64 {
	entry := o.Entry
	var errs []float64
	for r := 0; r < rounds; r++ {
		z := append([]float32(nil), src...)
		states := make([]uint16, len(src)/2)
		QuantizePairsPath(z, o, book, states)
		var cost float64
		for i, v := range src {
			d := float64(v - z[i])
			cost += d * d
		}
		errs = append(errs, cost/float64(len(src)))
		sum := make([]float64, len(book))
		cnt := make([]float64, len(book)/2)
		for i, s := range states {
			e := entry(s)
			sum[2*e] += float64(src[2*i])
			sum[2*e+1] += float64(src[2*i+1])
			cnt[e]++
		}
		for e := range cnt {
			if cnt[e] > 0 {
				book[2*e] = float32(sum[2*e] / cnt[e])
				book[2*e+1] = float32(sum[2*e+1] / cnt[e])
			}
		}
	}
	return errs
}

// EncodePairs writes one matrix in a pair tier: the same site vector, rotation
// and per-block normalisation as EncodeT4GAs, then the pair trellis, then each
// block's step fitted by least squares to the path.
func EncodePairs(w []float32, rows, cols int, q []float32, p GolemParams, kind nn.Quant) []byte {
	tier := nn.PairTierOf(kind)
	if cols%nn.T4GSeq != 0 {
		panic("compress: a pair-trellis row must be a multiple of 128 wide")
	}
	if p.ScaleBlock != 0 && p.ScaleBlock != nn.T4GBlock {
		panic("compress: a pair-trellis block is 64 weights and its step is not negotiable")
	}
	n := rows * cols
	rowBytes := tier.RowBytes(cols)
	out := make([]byte, rows*rowBytes)
	prep, norm := golemNormalise(w, rows, cols, q, p)

	states := make([]uint16, n/2)
	QuantizePairsPath(norm, PairOptsFor(kind), tier.Codebook(), states)

	nblk := n / nn.T4GBlock
	steps := golemFitSteps(prep, norm, nblk)
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			plane, codes := tier.Planes(out[r*rowBytes:(r+1)*rowBytes], cols)
			copy(plane, steps[r*cols/nn.T4GBlock:(r+1)*cols/nn.T4GBlock])
			for s := 0; s*nn.T4GSeq < cols; s++ {
				at := (r*cols + s*nn.T4GSeq) / 2
				tier.PutStates(codes[s*tier.SeqBytes:], states[at:at+nn.T4GSeq/2])
			}
		}
	})
	return out
}
