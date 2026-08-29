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
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// nearestInShell rounds x onto D4 and, if that lands past the shell, pulls x
// towards the origin until it does not. Only the point is kept: it has a code,
// and the block's own step multiplies it. The pull is a way of finding a
// representable point, not a scale anybody stores.
func nearestInShell(x, out []float32) {
	nearestDn(x, out)
	if coded(out) {
		return
	}
	var tmp [4]float32
	for s := float32(0.92); s > 0.05; s *= 0.92 {
		for i := range x {
			tmp[i] = x[i] * s
		}
		nearestDn(tmp[:], out)
		if coded(out) {
			return
		}
	}
	for i := range out {
		out[i] = 0
	}
}

func coded(p []float32) bool {
	return nn.D4Has([4]int8{int8(p[0]), int8(p[1]), int8(p[2]), int8(p[3])})
}

// D4Params is what the converter chose, and what the file then no longer needs
// to say.
type D4Params struct {
	Beta        float64 // how far a normalised block is scaled up before rounding
	ScaleBlock  int     // weights sharing one fp16 step; must be a multiple of 64
	HadGroup    int     // 0 leaves the matrix unrotated
	SearchScale bool
}

// chooseStep picks the one number a block of weights stores besides its codes.
// It is the block's only degree of freedom, and the search is what a plain
// multiple of the RMS leaves on the table.
//
// The search is over the codes themselves rather than over a handful of
// multiples of the RMS, because the codes are what can be written: a multiple
// that falls between two of them is rounded to one of them anyway, and two
// multiples that fall on the same code are one candidate tried twice. The span
// is a little over two octaves around the RMS, which is where the answer is for
// a block of outliers and for a flat one alike.
func chooseStep(blk []float32, lo, hi float32, beta float32, buf, pt []float32) float32 {
	var ss float64
	for _, v := range blk {
		ss += float64(v) * float64(v)
	}
	rms := float32(math.Sqrt(ss / float64(len(blk))))
	base := rms / beta
	first, last := nn.D4StepCode(base*lo), nn.D4StepCode(base*hi)
	best, bestStep := float32(math.MaxFloat32), float32(0)
	for c := int(first); c <= int(last); c++ {
		step := nn.D4Step(byte(c))
		if step <= 0 {
			continue
		}
		var err float32
		for i := 0; i*4 < len(blk); i++ {
			x := blk[i*4 : i*4+4]
			for j := range x {
				buf[j] = x[j] / step
			}
			nearestInShell(buf, pt)
			for j := 0; j < 4; j++ {
				d := x[j] - pt[j]*step
				err += d * d
			}
		}
		if err < best {
			best, bestStep = err, step
		}
	}
	if bestStep == 0 {
		// A block of zeros, or one no step could fit: the origin is a lattice
		// point and costs nothing to name.
		bestStep = nn.D4Step(0)
	}
	return bestStep
}

// EncodeD4G writes one matrix. q is the per-column vector the weights are
// scaled by — the reciprocal of what the activations will meet — or nil for a
// matrix that is neither scaled nor rotated. comp, when given, is the site's
// compensation plan; compress/gptq.go says what it does to the sweep.
func EncodeD4G(w []float32, rows, cols int, q []float32, p D4Params, comp *Comp) []byte {
	if cols%nn.D4Block != 0 {
		panic("compress: a D4G row must be a multiple of 64 wide")
	}
	sb := p.ScaleBlock
	if sb <= 0 || sb%nn.D4SubBlock != 0 {
		panic("compress: the scale block must be a multiple of 32")
	}
	rowBytes := cols / nn.D4Block * 26
	out := make([]byte, rows*rowBytes)

	// A block of outliers wants a coarse step and a flat one wants a fine step,
	// and neither is within a few percent of the other, so the span is wide.
	spanLo, spanHi := float32(1), float32(1)
	if p.SearchScale {
		spanLo, spanHi = 0.45, 2.4
	}
	beta := float32(p.Beta)

	Parallel(rows, func(lo, hi int) {
		row := make([]float32, cols)
		codes := make([]uint16, cols/4)
		steps := make([]byte, cols/nn.D4SubBlock)
		pt := make([]float32, 4)
		buf := make([]float32, 4)
		var e, corr [4]float32

		for r := lo; r < hi; r++ {
			copy(row, w[r*cols:(r+1)*cols])
			if q != nil {
				nn.PrepareD4G(row, q, p.HadGroup)
			}

			for k, a := 0, 0; a < cols; k++ {
				// One window: how far the compensation reaches. With no plan
				// the whole row is a single window and nothing propagates.
				m, R := cols-a, []float32(nil)
				if comp != nil {
					R, m = comp.block(k)
					if m%sb != 0 {
						R = nil
					}
				}
				for off := 0; off < m; off += sb {
					blk := row[a+off : a+off+sb]
					step := chooseStep(blk, spanLo, spanHi, beta, buf, pt)
					code := nn.D4StepCode(step)
					step = nn.D4Step(code)
					for i := 0; i*nn.D4SubBlock < sb; i++ {
						steps[(a+off)/nn.D4SubBlock+i] = code
					}
					for g := 0; g*4 < sb; g++ {
						col := off + g*4
						x := row[a+col : a+col+4]
						for j := range x {
							buf[j] = x[j] / step
						}
						nearestInShell(buf, pt)
						var p4 [4]int8
						for j := 0; j < 4; j++ {
							p4[j] = int8(pt[j])
						}
						c, ok := nn.D4Code(p4)
						if !ok {
							c, _ = nn.D4Code([4]int8{})
							pt[0], pt[1], pt[2], pt[3] = 0, 0, 0, 0
						}
						codes[(a+col)/4] = c
						if R == nil || col+4 >= m {
							continue
						}
						// The four columns just decided owe what they could not
						// represent to the ones after them, and R says in what
						// proportion. corr solves corr·R_gg = e forwards; the
						// rest of the window then takes corr·R_g,rest.
						for j := 0; j < 4; j++ {
							e[j] = x[j] - pt[j]*step
						}
						for i := 0; i < 4; i++ {
							s := e[i]
							for t := 0; t < i; t++ {
								s -= corr[t] * R[(col+t)*m+col+i]
							}
							if d := R[(col+i)*m+col+i]; d != 0 {
								corr[i] = s / d
							} else {
								corr[i] = 0
							}
						}
						rest := row[a+col+4 : a+m]
						for i := 0; i < 4; i++ {
							ci := corr[i]
							if ci == 0 {
								continue
							}
							ri := R[(col+i)*m+col+4 : (col+i)*m+m]
							for j := range rest {
								rest[j] -= ci * ri[j]
							}
						}
					}
				}
				a += m
			}

			scales, codes24 := nn.D4Planes(out[r*rowBytes:(r+1)*rowBytes], cols)
			copy(scales, steps)
			for b := 0; b*nn.D4Block < cols; b++ {
				nn.PutD4Codes(codes24[b*24:], codes[b*16:(b+1)*16])
			}
		}
	})
	return out
}

// RelErrD4G is what the codes cost the matrix they stand for, in the basis they
// were written in: ‖W-Ŵ‖/‖W‖ over the rotated, scaled weights. It is the
// quantizer's own error and nothing else's, which is what says whether there is
// room left in the quantizer or only in what surrounds it.
func RelErrD4G(w []float32, rows, cols int, q []float32, p D4Params, data []byte) float64 {
	stride := cols / nn.D4Block * 26
	var num, den float64
	row := make([]float32, cols)
	rec := make([]float32, cols)
	for r := 0; r < rows; r++ {
		copy(row, w[r*cols:(r+1)*cols])
		if q != nil {
			nn.PrepareD4G(row, q, p.HadGroup)
		}
		nn.DequantizeD4G(data[r*stride:(r+1)*stride], cols, rec)
		for j := range row {
			d := float64(row[j] - rec[j])
			num += d * d
			den += float64(row[j]) * float64(row[j])
		}
	}
	return math.Sqrt(num / den)
}
