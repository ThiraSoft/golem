package compress

// Turning a matrix into the bytes nn/d4g.go reads.
//
// The scheme is symmetric, and that is what makes it cheap at inference. Write
// q for the per-column vector the weights are scaled by and A for the rotation.
// The weights become A·(q ⊙ w) and the activations A·(x/q), and since AᵀA = I
// the product is unchanged — so the two sides go through the same function,
// nn.PrepareD4G, with reciprocal vectors.
//
// β does not survive into the file. The quantizer scales a block up by β to use
// the shell, rounds, and scales back; all the decoder ever sees is one fp16 —
// the block's step — and an integer point. So the format carries no resolution
// parameter, only the scale it produced.

import (
	"encoding/binary"
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// nearestInShell rounds x onto D4 and, if that lands past the shell, pulls x
// towards the origin until it does not. Only the point is kept: it is a shell
// point, so it has a code, and the block's own step multiplies it. The pull is
// a way of finding a representable point, not a scale anybody stores.
func nearestInShell(x, out []float32) {
	nearestDn(x, out)
	if norm2(out) <= nn.D4Radius {
		return
	}
	var tmp [4]float32
	for s := float32(0.92); s > 0.05; s *= 0.92 {
		for i := range x {
			tmp[i] = x[i] * s
		}
		nearestDn(tmp[:], out)
		if norm2(out) <= nn.D4Radius {
			return
		}
	}
	for i := range out {
		out[i] = 0
	}
}

// D4Params is what the converter chose, and what the file then no longer needs
// to say.
type D4Params struct {
	Beta        float64 // how far a normalised block is scaled up before rounding
	ScaleBlock  int     // weights sharing one fp16 step; must be a multiple of 64
	HadGroup    int     // 0 leaves the matrix unrotated
	SearchScale bool
}

// EncodeD4G writes one matrix. q is the per-column vector the weights are
// scaled by — the reciprocal of what the activations will meet — or nil for a
// matrix that is neither scaled nor rotated.
func EncodeD4G(w []float32, rows, cols int, q []float32, p D4Params) []byte {
	if cols%nn.D4Block != 0 {
		panic("compress: a D4G row must be a multiple of 64 wide")
	}
	sb := p.ScaleBlock
	if sb <= 0 || sb%nn.D4Block != 0 {
		panic("compress: the scale block must be a multiple of 64")
	}
	rowBytes := cols / nn.D4Block * 26
	out := make([]byte, rows*rowBytes)

	mults := []float32{1}
	if p.SearchScale {
		// The step is the block's only degree of freedom, so the search around
		// its RMS is wide: a block of outliers wants a coarse step and a flat
		// one wants a fine step, and neither is within a few percent of the
		// other.
		mults = []float32{0.62, 0.72, 0.82, 0.91, 1, 1.1, 1.22, 1.38, 1.6, 1.9}
	}
	beta := float32(p.Beta)

	Parallel(rows, func(lo, hi int) {
		row := make([]float32, cols)
		bestCodes := make([]uint16, sb/4)
		codes := make([]uint16, sb/4)
		pt := make([]float32, 4)
		buf := make([]float32, 4)

		for r := lo; r < hi; r++ {
			copy(row, w[r*cols:(r+1)*cols])
			if q != nil {
				nn.PrepareD4G(row, q, p.HadGroup)
			}
			scales, codes24 := nn.D4Planes(out[r*rowBytes:(r+1)*rowBytes], cols)

			for b := 0; b*sb < cols; b++ {
				blk := row[b*sb : (b+1)*sb]
				var ss float64
				for _, v := range blk {
					ss += float64(v) * float64(v)
				}
				rms := float32(math.Sqrt(ss / float64(sb)))

				bestErr, bestStep := float32(math.MaxFloat32), float32(0)
				for _, mu := range mults {
					step := Fp16round(rms * mu / beta)
					if step <= 0 {
						continue
					}
					var err float32
					bad := false
					for i := 0; i*4 < sb; i++ {
						x := blk[i*4 : i*4+4]
						for j := range x {
							buf[j] = x[j] / step
						}
						nearestInShell(buf, pt)
						var p4 [4]int8
						for j := 0; j < 4; j++ {
							p4[j] = int8(pt[j])
							d := x[j] - pt[j]*step
							err += d * d
						}
						c, ok := nn.D4Code(p4)
						if !ok {
							bad = true
							break
						}
						codes[i] = c
					}
					if !bad && err < bestErr {
						bestErr, bestStep = err, step
						copy(bestCodes, codes)
					}
				}
				if bestStep == 0 {
					// A block of zeros, or one no step could fit: the origin is
					// a lattice point and costs nothing to name.
					bestStep = Fp16round(1)
					zero, _ := nn.D4Code([4]int8{})
					for i := range bestCodes {
						bestCodes[i] = zero
					}
				}
				first := b * (sb / nn.D4Block)
				for k := 0; k*nn.D4Block < sb; k++ {
					blk := first + k
					binary.LittleEndian.PutUint16(scales[blk*2:], nn.FloatToHalf(bestStep))
					nn.PutD4Codes(codes24[blk*24:], bestCodes[k*16:(k+1)*16])
				}
			}
		}
	})
	return out
}
