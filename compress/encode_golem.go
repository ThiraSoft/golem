package compress

// Turning a matrix into the bytes nn/pair.go reads.
//
// The scheme around the codes is Golem's and unchanged — the per-column vector,
// the Hadamard rotation, one step per sixty-four weights — because none of it
// was ever about the codebook. What changes is the middle: where the lattice
// searched a step per block and then rounded four weights at a time, this runs
// a Viterbi over whole sequences of 128 and only then asks what step fits the
// path it found.
//
// The order is not a preference. A block's RMS is the scale that makes it unit
// variance, which is not the scale that reconstructs it best; the lattice buys
// that difference with a grid of candidate steps per block and a trellis
// cannot, because one path spans two blocks and the search would have to be
// joint. Least squares takes it exactly and for nothing, once the path exists,
// and compress/codec.go measures it at about four percent of the error.
//
// The number that normalises a block on the way in is therefore not the number
// the file stores, and it is not rounded to anything: rounding it would move
// the path, which on the eight-bit grid means a four percent gain error on a
// codebook that was measured to want none.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// GolemParams is what the converter chose, and what the file then no longer
// needs to say. The name is the scheme's, not the lattice's: ScaleBlock and
// HadGroup describe the per-column vector and the rotation, which the trellis
// shares with everything golem's own format has ever written. There is no
// scale-search parameter here — a lattice's step was a fraction of its block's
// RMS, tried at a few candidate multiples because the block was scaled up into
// a shell; a trellis step is the block's RMS itself, fitted by least squares
// after the path is chosen, which is a question the codebook answers and not
// one this struct has anything to say about.
type GolemParams struct {
	ScaleBlock int // weights sharing one step code; must be a multiple of 32
	HadGroup   int // 0 leaves the matrix unrotated
}

// EncodeGolem writes one matrix in a pair tier. q is the per-column vector the
// weights are scaled by — the reciprocal of what the activations will meet — or
// nil for a matrix that is neither scaled nor rotated.
//
// The whole chunk goes through the trellis in one call rather than a row at a
// time, because the encoder that matters is on a card and a pass of sixteen
// million weights is what pays for the round trip. compress.PairAccel is where
// it hooks in, and the processor takes any shape the card refuses.
func EncodeGolem(w []float32, rows, cols int, q []float32, p GolemParams, kind nn.Quant) []byte {
	if nn.PairTierOf(kind) == nil {
		panic("compress: " + kind.String() + " is not one of golem's formats")
	}
	return EncodePairs(w, rows, cols, q, p, kind)
}

// golemNormalise puts a matrix in the basis it is coded in — the site's vector,
// then the rotation — and returns that, prep, beside norm, the same weights at
// unit RMS per block, which is what the codebook is built to meet. prep is kept
// because the step is fitted against it once the path exists.
func golemNormalise(w []float32, rows, cols int, q []float32, p GolemParams) (prep, norm []float32) {
	n := rows * cols
	// The weights in the basis they are coded in: the site's vector, then the
	// rotation. Kept, because the step is fitted against them at the end.
	prep = make([]float32, n)
	Parallel(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := prep[r*cols : (r+1)*cols]
			copy(row, w[r*cols:(r+1)*cols])
			if q != nil {
				nn.PrepareGolem(row, q, p.HadGroup)
			}
		}
	})

	// Unit variance per block, so that the codebook — which is N(0,1) and the
	// same for every tensor — meets a source of the size it was built for.
	norm = make([]float32, n)
	nblk := n / nn.GolemBlock
	Parallel(nblk, func(lo, hi int) {
		for b := lo; b < hi; b++ {
			blk := prep[b*nn.GolemBlock : (b+1)*nn.GolemBlock]
			var ss float64
			for _, v := range blk {
				ss += float64(v) * float64(v)
			}
			inv := float32(0)
			if rms := float32(math.Sqrt(ss / float64(nn.GolemBlock))); rms > 0 {
				inv = 1 / rms
			}
			for i, v := range blk {
				norm[b*nn.GolemBlock+i] = v * inv
			}
		}
	})

	return prep, norm
}

// golemFitSteps is each block's step, fitted by least squares to what the path
// reconstructs at unit scale and rounded onto the eight-bit grid.
func golemFitSteps(prep, norm []float32, nblk int) []byte {
	// The step, per block, fitted to the path and then rounded onto the grid
	// the eight bits name.
	steps := make([]byte, nblk)
	Parallel(nblk, func(lo, hi int) {
		for b := lo; b < hi; b++ {
			var num, den float64
			for i := b * nn.GolemBlock; i < (b+1)*nn.GolemBlock; i++ {
				num += float64(prep[i]) * float64(norm[i])
				den += float64(norm[i]) * float64(norm[i])
			}
			if den > 0 {
				steps[b] = nn.GolemStepCode(float32(num / den))
			}
		}
	})

	return steps
}

// RelErr is what the codes cost the matrix they stand for, in the basis they
// were written in: ‖W-Ŵ‖/‖W‖ over the rotated, scaled weights. It is the
// quantizer's own error and nothing else's, which is what says whether there is
// room left in the quantizer or only in what surrounds it. kind says which of
// golem's own formats wrote the bytes.
func RelErr(w []float32, rows, cols int, q []float32, p GolemParams, data []byte, kind nn.Quant) float64 {
	m := nn.Matrix{Data: data, Quant: kind, Rows: rows, Cols: cols}
	var num, den float64
	row := make([]float32, cols)
	rec := make([]float32, cols)
	for r := 0; r < rows; r++ {
		copy(row, w[r*cols:(r+1)*cols])
		if q != nil {
			nn.PrepareGolem(row, q, p.HadGroup)
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

// EnergyGolem is what a matrix's quantization costs the product it sits in,
// relative to what the product is: ‖(W-Ŵ)X‖² over ‖WX‖², estimated from the
// activations the accumulator saw. kind says which of golem's own formats wrote
// the bytes.
//
// The difference is measured in the basis the weights were written in and then
// carried back to the one the activations were measured in, because that is
// where the Hessian lives. Which is also why pre is needed and not just q: the
// two are reciprocal, and undoing a rotation is not the same as applying it.
func EnergyGolem(w []float32, rows, cols int, q, pre []float32, p GolemParams, data []byte, kind nn.Quant, a *Acc) (float64, float64) {
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
				nn.PrepareGolem(row, q, p.HadGroup)
			}
			m.Row(r, rec)
			for j := range d {
				d[j] = row[j] - rec[j]
			}
			if pre != nil {
				nn.UnprepareGolem(d, pre, p.HadGroup)
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
