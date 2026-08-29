package compress

// Turning a matrix into the bytes nn/l8g.go reads.
//
// The same sweep as the lattice's, with the codebook swapped: a weight is
// divided by its block's step and named by the nearest of Lloyd's eight levels,
// where the lattice named four weights at once by the nearest point of a shell.
// Everything around it — the per-column vector, the rotation, the step per
// thirty-two weights and the search that chooses it — is unchanged, because
// none of it was ever about the codebook.
//
// The step search is where the difference shows. A lattice has to be asked
// whether a point is representable at all and pulled back towards the origin
// when it is not; eight levels have no outside, so a value past the last one
// clamps to it and the whole shell apparatus goes away.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// EncodeL8G writes one matrix in the eight-level format. The arguments are
// EncodeD4G's and mean the same things.
func EncodeL8G(w []float32, rows, cols int, q []float32, p D4Params) []byte {
	if cols%nn.L8Block != 0 {
		panic("compress: an L8G row must be a multiple of 64 wide")
	}
	sb := p.ScaleBlock
	if sb <= 0 || sb%nn.D4SubBlock != 0 {
		panic("compress: the scale block must be a multiple of 32")
	}
	rowBytes := cols / nn.L8Block * 26
	out := make([]byte, rows*rowBytes)

	spanLo, spanHi := float32(1), float32(1)
	if p.SearchScale {
		spanLo, spanHi = 0.45, 2.4
	}
	beta := float32(p.Beta)

	Parallel(rows, func(lo, hi int) {
		row := make([]float32, cols)
		codes := make([]byte, cols)
		steps := make([]byte, cols/nn.D4SubBlock)

		for r := lo; r < hi; r++ {
			copy(row, w[r*cols:(r+1)*cols])
			if q != nil {
				nn.PrepareD4G(row, q, p.HadGroup)
			}
			for off := 0; off < cols; off += sb {
				blk := row[off : off+sb]
				step := chooseStepL8(blk, spanLo, spanHi, beta)
				code := nn.D4StepCode(step)
				step = nn.D4Step(code)
				for i := 0; i*nn.D4SubBlock < sb; i++ {
					steps[off/nn.D4SubBlock+i] = code
				}
				for i, v := range blk {
					codes[off+i] = nn.L8Code(v / step)
				}
			}
			plane, run := nn.L8Planes(out[r*rowBytes:(r+1)*rowBytes], cols)
			copy(plane, steps)
			for b := 0; b*nn.L8Block < cols; b++ {
				nn.PutL8Codes(run[b*24:], codes[b*nn.L8Block:(b+1)*nn.L8Block])
			}
		}
	})
	return out
}

// chooseStepL8 is chooseStep for a codebook with no outside: every candidate
// step is representable, so the search is the error alone and the early exit
// is the only thing keeping it cheap.
func chooseStepL8(blk []float32, lo, hi, beta float32) float32 {
	var ss float64
	for _, v := range blk {
		ss += float64(v) * float64(v)
	}
	rms := float32(math.Sqrt(ss / float64(len(blk))))
	base := rms / beta
	first, last := int(nn.D4StepCode(base*lo)), int(nn.D4StepCode(base*hi))
	best, bestStep := float32(math.MaxFloat32), float32(0)
	try := func(c int) {
		step := nn.D4Step(byte(c))
		if step <= 0 {
			return
		}
		var err float32
		for _, v := range blk {
			d := v - nn.L8Level(nn.L8Code(v/step))*step
			err += d * d
			if err >= best {
				return
			}
		}
		if err < best {
			best, bestStep = err, step
		}
	}
	if mid := int(nn.D4StepCode(base)); mid >= first && mid <= last {
		try(mid)
	}
	for c := first; c <= last; c++ {
		try(c)
	}
	if bestStep == 0 {
		bestStep = nn.D4Step(0)
	}
	return bestStep
}
