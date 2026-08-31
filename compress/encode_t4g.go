package compress

// Turning a matrix into the bytes nn/t4g.go reads.
//
// The scheme around the codes is D4G's and unchanged — the per-column vector,
// the Hadamard rotation, one step per sixty-four weights — because none of it
// was ever about the codebook. What changes is the middle: where the lattice
// searched a step per block and then rounded four weights at a time, this runs
// a Viterbi over whole sequences of 128 and only then asks what step fits the
// path it found.
//
// The order is not a preference. A block's RMS is the scale that makes it unit
// variance, which is not the scale that reconstructs it best; the lattice buys
// that difference with a grid of candidate steps per block and a trellis cannot,
// because one path spans two blocks and the search would have to be joint.
// Least squares takes it exactly and for nothing, once the path exists, and
// compress/codec.go measures it at about four percent of the error.
//
// The number that normalises a block on the way in is therefore not the number
// the file stores, and it is not rounded to anything: rounding it would move
// the path, which on the eight-bit grid means a four percent gain error on a
// codebook that was measured to want none.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// D4Params is what the converter chose, and what the file then no longer needs
// to say. The name is the scheme's, not the lattice's: ScaleBlock and HadGroup
// describe the per-column vector and the rotation, which the trellis shares
// with everything golem's own format has ever written. There is no scale-search
// parameter here — a lattice's step was a fraction of its block's RMS, tried at
// a few candidate multiples because the block was scaled up into a shell; a
// trellis step is the block's RMS itself, fitted by least squares after the
// path is chosen, which is a question the codebook answers and not one this
// struct has anything to say about.
type D4Params struct {
	ScaleBlock int // weights sharing one step code; must be a multiple of 32
	HadGroup   int // 0 leaves the matrix unrotated
}

// T4GOpts is the trellis the file is written with. There is nothing to choose:
// the sequence length, the rate and the state width are all fixed by the format
// and by what a workgroup's sixty-four kibibytes hold, and the codebook's gain
// is one because any departure from one costs — measured, and recorded among
// the closed questions.
func T4GOpts() TrellisOpts { return T4GOptsFor(nn.T4G) }

// T4GOptsFor is the same for whichever tier.
func T4GOptsFor(q nn.Quant) TrellisOpts {
	k := nn.T4GK
	switch q {
	case nn.T5G:
		k = nn.T5GK
	case nn.T3G:
		k = nn.T3GK
	}
	return TrellisOpts{K: k, L: nn.T4GL, Seq: nn.T4GSeq, Gain: 1, Code: Code1MAD}
}

// EncodeT4G writes one matrix. q is the per-column vector the weights are
// scaled by — the reciprocal of what the activations will meet — or nil for a
// matrix that is neither scaled nor rotated.
//
// The whole chunk goes through the trellis in one call rather than a row at a
// time, because the encoder that matters is on a card and a pass of sixteen
// million weights is what pays for the round trip. compress.TrellisPathAccel is
// where it hooks in, and the processor takes any shape the card refuses.
func EncodeT4G(w []float32, rows, cols int, q []float32, p D4Params) []byte {
	return EncodeT4GAs(w, rows, cols, q, p, nn.T4G)
}

// EncodeT4GAs is the same in whichever tier. The wide one is for the logit
// head and nothing else: a bit a weight over a tenth of a model is a tenth of
// a bit, and it is the tensor that makes the logits rather than one whose
// error the layers after it absorb.
func EncodeT4GAs(w []float32, rows, cols int, q []float32, p D4Params, kind nn.Quant) []byte {
	if cols%nn.T4GSeq != 0 {
		panic("compress: a T4G row must be a multiple of 128 wide")
	}
	if p.ScaleBlock != 0 && p.ScaleBlock != nn.T4GBlock {
		panic("compress: a T4G block is 64 weights and its step is not negotiable")
	}
	n := rows * cols
	rowBytes := nn.T4GRowBytesN(cols, kind)
	seqBytes := nn.T4GSeqBytesN(kind)
	out := make([]byte, rows*rowBytes)

	// The weights in the basis they are coded in: the site's vector, then the
	// rotation. Kept, because the step is fitted against them at the end.
	prep := make([]float32, n)
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := prep[r*cols : (r+1)*cols]
			copy(row, w[r*cols:(r+1)*cols])
			if q != nil {
				nn.PrepareD4G(row, q, p.HadGroup)
			}
		}
	})

	// Unit variance per block, so that the codebook — which is N(0,1) and the
	// same for every tensor — meets a source of the size it was built for.
	norm := make([]float32, n)
	nblk := n / nn.T4GBlock
	Parallel(nblk, func(lo, hi int) {
		for b := lo; b < hi; b++ {
			blk := prep[b*nn.T4GBlock : (b+1)*nn.T4GBlock]
			var ss float64
			for _, v := range blk {
				ss += float64(v) * float64(v)
			}
			inv := float32(0)
			if rms := float32(math.Sqrt(ss / float64(nn.T4GBlock))); rms > 0 {
				inv = 1 / rms
			}
			for i, v := range blk {
				norm[b*nn.T4GBlock+i] = v * inv
			}
		}
	})

	// The path. norm comes back holding what it reconstructs, at unit scale.
	states := make([]uint16, n)
	QuantizeTrellisPath(norm, T4GOptsFor(kind), states)

	// The step, per block, fitted to the path and then rounded onto the grid
	// the eight bits name.
	steps := make([]byte, nblk)
	Parallel(nblk, func(lo, hi int) {
		for b := lo; b < hi; b++ {
			var num, den float64
			for i := b * nn.T4GBlock; i < (b+1)*nn.T4GBlock; i++ {
				num += float64(prep[i]) * float64(norm[i])
				den += float64(norm[i]) * float64(norm[i])
			}
			if den > 0 {
				steps[b] = nn.T4GStepCode(float32(num / den))
			}
		}
	})

	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			plane, codes := nn.T4GPlanesN(out[r*rowBytes:(r+1)*rowBytes], cols, kind)
			copy(plane, steps[r*cols/nn.T4GBlock:(r+1)*cols/nn.T4GBlock])
			for s := 0; s*nn.T4GSeq < cols; s++ {
				at := r*cols + s*nn.T4GSeq
				nn.PutT4GStatesN(codes[s*seqBytes:], states[at:at+nn.T4GSeq], kind)
			}
		}
	})
	return out
}

// RelErr is what the codes cost the matrix they stand for, in the basis they
// were written in: ‖W-Ŵ‖/‖W‖ over the rotated, scaled weights. It is the
// quantizer's own error and nothing else's, which is what says whether there is
// room left in the quantizer or only in what surrounds it. kind says which of
// golem's own formats wrote the bytes.
func RelErr(w []float32, rows, cols int, q []float32, p D4Params, data []byte, kind nn.Quant) float64 {
	m := nn.Matrix{Data: data, Quant: kind, Rows: rows, Cols: cols}
	var num, den float64
	row := make([]float32, cols)
	rec := make([]float32, cols)
	for r := 0; r < rows; r++ {
		copy(row, w[r*cols:(r+1)*cols])
		if q != nil {
			nn.PrepareD4G(row, q, p.HadGroup)
		}
		m.Row(r, rec)
		for j := range row {
			d := float64(row[j] - rec[j])
			num += d * d
			den += float64(row[j]) * float64(row[j])
		}
	}
	return math.Sqrt(num / den)
}

// EnergyD4G is what a matrix's quantization costs the product it sits in,
// relative to what the product is: ‖(W-Ŵ)X‖² over ‖WX‖², estimated from the
// activations the accumulator saw. kind says which of golem's own formats wrote
// the bytes.
//
// The difference is measured in the basis the weights were written in and then
// carried back to the one the activations were measured in, because that is
// where the Hessian lives. Which is also why pre is needed and not just q: the
// two are reciprocal, and undoing a rotation is not the same as applying it.
func EnergyD4G(w []float32, rows, cols int, q, pre []float32, p D4Params, data []byte, kind nn.Quant, a *Acc) (float64, float64) {
	m := nn.Matrix{Data: data, Quant: kind, Rows: rows, Cols: cols}
	nums := make([]float64, rows)
	dens := make([]float64, rows)
	Parallel(rows, func(lo, hi int) {
		row := make([]float32, cols)
		rec := make([]float32, cols)
		d := make([]float32, cols)
		for r := lo; r < hi; r++ {
			copy(row, w[r*cols:(r+1)*cols])
			if q != nil {
				nn.PrepareD4G(row, q, p.HadGroup)
			}
			m.Row(r, rec)
			for j := range d {
				d[j] = row[j] - rec[j]
			}
			if pre != nil {
				nn.UnprepareD4G(d, pre, p.HadGroup)
			}
			nums[r] = a.Energy(d)
			copy(d, w[r*cols:(r+1)*cols])
			dens[r] = a.Energy(d)
		}
	})
	var num, den float64
	for r := range nums {
		num += nums[r]
		den += dens[r]
	}
	return num, den
}
