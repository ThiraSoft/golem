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
func nearestInShell(x, out []float32, bits int) {
	nearestDn(x, out)
	if coded(out, bits) {
		return
	}
	// Past the shell. Pulling x towards the origin lands on points that are
	// coded, but the first one that fits is not the closest one that fits —
	// the ray does not pass through the best point of a lattice — so a handful
	// of pulls are tried and the nearest of them wins.
	//
	// The handful starts where the geometry says it should: the pull that puts
	// x exactly on the shell. Scanning down from one instead would be a dozen
	// wasted roundings for a subvector far outside, and the step search asks
	// for this on every subvector of every candidate step, which is where an
	// encoder's afternoon goes.
	var tmp, best [4]float32
	bestD := float32(math.MaxFloat32)
	full, _ := nn.D4TierNorms(bits)
	s0 := float32(math.Sqrt(float64(full) / float64(norm2(x))))
	if s0 > 1 {
		s0 = 1
	}
	for k := 0; k < 10; k++ {
		s := s0 * (1 - 0.04*float32(k))
		for i := range x {
			tmp[i] = x[i] * s
		}
		nearestDn(tmp[:], out)
		if !coded(out, bits) {
			continue
		}
		var d float32
		for i := range x {
			e := x[i] - out[i]
			d += e * e
		}
		if d < bestD {
			bestD = d
			copy(best[:], out)
		}
	}
	if bestD == float32(math.MaxFloat32) {
		for i := range out {
			out[i] = 0
		}
		return
	}
	copy(out, best[:])
}

func coded(p []float32, bits int) bool {
	return nn.D4Has([4]int8{int8(p[0]), int8(p[1]), int8(p[2]), int8(p[3])}, bits)
}

// D4Params is what the converter chose, and what the file then no longer needs
// to say.
type D4Params struct {
	Beta        float64 // how far a normalised block is scaled up before rounding
	ScaleBlock  int     // weights sharing one step code; must be a multiple of 32
	HadGroup    int     // 0 leaves the matrix unrotated
	Bits        int     // code width; zero means the ordinary twelve
	SearchScale bool
}

// Width is the code width the parameters ask for.
func (p D4Params) Width() int {
	if p.Bits == 0 {
		return nn.D4Bits
	}
	return p.Bits
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
func chooseStep(blk []float32, lo, hi float32, beta float32, buf, pt []float32, bits int) float32 {
	var ss float64
	for _, v := range blk {
		ss += float64(v) * float64(v)
	}
	rms := float32(math.Sqrt(ss / float64(len(blk))))
	base := rms / beta
	first, last := int(nn.D4StepCode(base*lo)), int(nn.D4StepCode(base*hi))
	best, bestStep := float32(math.MaxFloat32), float32(0)
	// The RMS itself first, so that the bound below has something to work with
	// from the start: it is where the answer usually is, and every candidate
	// after it can be abandoned as soon as it is worse.
	try := func(c int) {
		step := nn.D4Step(byte(c))
		if step <= 0 {
			return
		}
		var err float32
		for i := 0; i*4 < len(blk); i++ {
			x := blk[i*4 : i*4+4]
			for j := range x {
				buf[j] = x[j] / step
			}
			nearestInShell(buf, pt, bits)
			for j := 0; j < 4; j++ {
				d := x[j] - pt[j]*step
				err += d * d
			}
			if err >= best {
				// Already worse than something we have, and the error only
				// grows. A step far from the right one is abandoned after two
				// or three subvectors instead of thirty-two.
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
	bits := p.Width()
	run := nn.D4Block / 4 * bits / 8
	rowBytes := cols / nn.D4Block * nn.D4BlockBytes(bits)
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
					step := chooseStep(blk, spanLo, spanHi, beta, buf, pt, bits)
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
						nearestInShell(buf, pt, bits)
						var p4 [4]int8
						for j := 0; j < 4; j++ {
							p4[j] = int8(pt[j])
						}
						c, ok := nn.D4CodeN(p4, bits)
						if !ok {
							c, _ = nn.D4CodeN([4]int8{}, bits)
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

			scales, run24 := nn.D4PlanesN(out[r*rowBytes:(r+1)*rowBytes], cols, bits)
			copy(scales, steps)
			for b := 0; b*nn.D4Block < cols; b++ {
				nn.PutD4CodesN(run24[b*run:], codes[b*16:(b+1)*16], bits)
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
	stride := cols / nn.D4Block * nn.D4BlockBytes(p.Width())
	var num, den float64
	row := make([]float32, cols)
	rec := make([]float32, cols)
	for r := 0; r < rows; r++ {
		copy(row, w[r*cols:(r+1)*cols])
		if q != nil {
			nn.PrepareD4G(row, q, p.HadGroup)
		}
		nn.DequantizeD4GN(data[r*stride:(r+1)*stride], cols, p.Width(), rec)
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
// activations the accumulator saw.
//
// The difference is measured in the basis the weights were written in and then
// carried back to the one the activations were measured in, because that is
// where the Hessian lives. Which is also why pre is needed and not just q: the
// two are reciprocal, and undoing a rotation is not the same as applying it.
func EnergyD4G(w []float32, rows, cols int, q, pre []float32, p D4Params, data []byte, a *Acc) (float64, float64) {
	stride := cols / nn.D4Block * nn.D4BlockBytes(p.Width())
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
			nn.DequantizeD4GN(data[r*stride:(r+1)*stride], cols, p.Width(), rec)
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
