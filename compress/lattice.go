package compress

// Lattice quantization: the codebook nobody has to store.
//
// A trained codebook costs a nearest-neighbour search against every one of its
// entries — a quarter of a million multiplies per weight at k=256, which no
// four-billion-parameter model will sit through. A lattice replaces the search
// with arithmetic: the nearest point of E8 (or of D4) is found by rounding,
// with one correction, in time that does not depend on how many points the
// codebook has.
//
// E8 is the densest packing in eight dimensions, so at equal rate it loses less
// than any trained codebook of the same size is likely to — and it is what
// QuIP# quantizes onto. The rate is set by the shell: only the points within a
// squared radius are addressable, and the index is log2 of how many there are.

import (
	"math"
)

// Lattice names one of the two we use.
type Lattice int

const (
	LatE8 Lattice = iota // dimension 8
	LatD4                // dimension 4
)

func (l Lattice) Dim() int {
	if l == LatD4 {
		return 4
	}
	return 8
}

func (l Lattice) String() string {
	if l == LatD4 {
		return "D4"
	}
	return "E8"
}

// nearestDn rounds onto Dn — the integer points of even coordinate sum — by
// rounding each coordinate and, if the sum came out odd, moving the one
// coordinate that was rounded the furthest.
func nearestDn(x, out []float32) {
	sum := 0
	worst, wdist := 0, float32(-1)
	for i, v := range x {
		r := float32(math.Round(float64(v)))
		out[i] = r
		sum += int(r)
		if d := float32(math.Abs(float64(v - r))); d > wdist {
			worst, wdist = i, d
		}
	}
	if sum&1 != 0 {
		if x[worst] > out[worst] {
			out[worst]++
		} else {
			out[worst]--
		}
	}
}

// nearestE8 gives the closest point of E8 = D8 ∪ (D8 + ½), and its distance.
func nearestE8(x, out, tmp []float32) float32 {
	nearestDn(x, out)
	d0 := sqdist(x, out)
	for i := range x {
		tmp[i] = x[i] - 0.5
	}
	nearestDn(tmp, tmp)
	for i := range tmp {
		tmp[i] += 0.5
	}
	if d1 := sqdist(x, tmp); d1 < d0 {
		copy(out, tmp)
		return d1
	}
	return d0
}

func sqdist(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func norm2(a []float32) float32 {
	var s float32
	for _, v := range a {
		s += v * v
	}
	return s
}

// quantizeLattice puts x on the nearest lattice point inside the shell,
// shrinking x towards the origin when the nearest point falls outside it. The
// result is written back over x.
func quantizeLattice(l Lattice, x, out, tmp []float32, maxNorm2 float32) {
	scale := float32(1)
	for try := 0; try < 24; try++ {
		for i := range x {
			tmp[i] = x[i] * scale
		}
		if l == LatD4 {
			nearestDn(tmp, out)
		} else {
			t2 := make([]float32, len(x))
			nearestE8(tmp, out, t2)
		}
		if norm2(out) <= maxNorm2 {
			break
		}
		scale *= 0.9
	}
	for i := range out {
		x[i] = out[i] / scale
	}
}

// shellSize counts the lattice points within a squared radius, which is what
// the index must address. Both counts are classical: E8 has 240·σ₃(n) vectors
// of norm 2n, D4 has 24 times the sum of the odd divisors of n.
func shellSize(l Lattice, maxNorm2 float32) int {
	total := 1
	for n := 1; float32(2*n) <= maxNorm2+1e-6; n++ {
		if l == LatE8 {
			total += 240 * sigma3(n)
		} else {
			total += 24 * sigmaOdd(n)
		}
	}
	return total
}

func sigma3(n int) int {
	s := 0
	for d := 1; d*d <= n; d++ {
		if n%d == 0 {
			s += d * d * d
			if e := n / d; e != d {
				s += e * e * e
			}
		}
	}
	return s
}

func sigmaOdd(n int) int {
	s := 0
	for d := 1; d <= n; d++ {
		if n%d == 0 && d%2 == 1 {
			s += d
		}
	}
	return s
}

// The box: D4 without a table.
//
// A shell is the densest set of lattice points of a given count, which is why
// it quantizes best — but it is an arbitrary set, so a code is an index into an
// enumeration and decoding one means a table lookup. Constraining each
// coordinate to a fixed range instead gives a set a shader decodes with shifts
// alone: four coordinates of three bits, and D4's even-sum rule recovers the
// fourth's low bit, so 8⁴/2 = 2048 points fit in eleven bits and no table is
// read at all.
//
// It packs a little worse than the shell of the same size. Whether that costs
// anything the model notices is a question for the corpus.
const boxLow, boxHigh = -4, 3

// quantizeBox puts x on the nearest D4 point whose coordinates all lie within
// the box, writing the result back over x.
func quantizeBox(x, out []float32) {
	nearestDn(x, out)
	// Rounding first and clamping after can leave an odd sum, so a clamp that
	// broke the parity is repaired the way nearestDn repairs its own: move the
	// coordinate that was rounded furthest, among those still free to move.
	sum := 0
	clamped := false
	for i, v := range out {
		if v < boxLow {
			v, clamped = boxLow, true
		} else if v > boxHigh {
			v, clamped = boxHigh, true
		}
		out[i] = v
		sum += int(v)
	}
	if clamped && sum&1 != 0 {
		worst, wdist := -1, float32(-1)
		for i := range x {
			up := x[i] > out[i]
			if (up && out[i] >= boxHigh) || (!up && out[i] <= boxLow) {
				continue
			}
			if d := absf(x[i] - out[i]); d > wdist {
				worst, wdist = i, d
			}
		}
		if worst >= 0 {
			if x[worst] > out[worst] {
				out[worst]++
			} else {
				out[worst]--
			}
		}
	}
	copy(x, out)
}

func absf(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// boxSize is how many points the box holds: every assignment of the four
// coordinates whose sum is even, which is exactly half of them.
func boxSize() int {
	span := boxHigh - boxLow + 1
	return span * span * span * span / 2
}
