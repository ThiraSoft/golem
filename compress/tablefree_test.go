package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// What a codebook computed from the code costs against one read from a table.
//
// The table is the only memory the D4G kernel touches that is not weights, and
// it is 11 to 17% of a dispatch. A code that could be turned into a point by
// arithmetic alone would take that back — and there is an exact way to do it.
// D4 is the integers of even sum, so three coordinates of three bits and a
// fourth of three bits whose low bit is the parity of the others is a bijection
// from twelve bits onto 4096 lattice points, decoded in shifts and no lookup.
//
// The point set it gives is a box, and the shell is a ball. This measures what
// that costs, on the source the rotation makes: four-dimensional Gaussian
// noise, each codebook at its own best scale.
func TestTableFreeCodebookCosts(t *testing.T) {
	const n = 200000
	r := rand.New(rand.NewSource(19))
	x := make([][4]float32, n)
	for i := range x {
		for j := 0; j < 4; j++ {
			x[i][j] = float32(r.NormFloat64())
		}
	}
	shell := bestScale(x, func(v, out []float32) {
		nearestDn(v, out)
		for norm2(out) > float32(nn.D4Radius) {
			for j := range v {
				v[j] *= 0.97
			}
			nearestDn(v, out)
		}
	})
	box := bestScale(x, func(v, out []float32) {
		nearestDn(v, out)
		// The box the arithmetic decoder can name: three coordinates in
		// [-4,3], and a fourth on the even numbers of [-8,6] with the parity
		// of the others added back.
		sum := 0
		for j := 0; j < 3; j++ {
			out[j] = clampF(out[j], -4, 3)
			sum += int(out[j])
		}
		q := math.Round(float64(out[3]-float32(sum&1)) / 2)
		q = math.Max(-4, math.Min(3, q))
		out[3] = float32(2*q) + float32(sum&1)
	})
	t.Logf("4096 points: %.2f dB from a shell in a table, %.2f dB from a box in arithmetic — the table buys %.2f dB",
		shell, box, shell-box)
	if shell <= box {
		t.Errorf("the box is no worse than the shell, which would make the table pure cost")
	}
}

func clampF(v float32, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// bestScale is the signal-to-noise a codebook reaches on this sample, at
// whichever step serves it best — the two have different best steps and
// comparing them at one would measure the step.
func bestScale(x [][4]float32, quant func(v, out []float32)) float64 {
	best := 0.0
	v := make([]float32, 4)
	out := make([]float32, 4)
	for mu := 0.15; mu < 1.2; mu *= 1.06 {
		var num, den float64
		for i := range x {
			for j := 0; j < 4; j++ {
				v[j] = x[i][j] / float32(mu)
				den += float64(x[i][j]) * float64(x[i][j])
			}
			quant(v, out)
			for j := 0; j < 4; j++ {
				d := float64(x[i][j]) - float64(out[j])*mu
				num += d * d
			}
		}
		if snr := -10 * math.Log10(num/den); snr > best {
			best = snr
		}
	}
	return best
}

// And the codebook that needs no table at all, because it has eight entries.
//
// Applying a curve to a uniform code — companding — gives a non-uniform scalar
// quantizer, and the best one for a Gaussian at a given rate is Lloyd's. At
// three bits that is eight levels, which fit in registers: no lookup, no
// memory, nothing for the kernel to read but weights.
//
// So the question the lattice has to answer is not whether it beats a box. It
// is whether four dimensions beat one at the same rate, against the best one
// dimension can do — which is what a k-quant is, and is why the gap between
// this format and llama.cpp's is not larger than it is.
func TestLatticeAgainstTheBestScalar(t *testing.T) {
	const n = 200000
	r := rand.New(rand.NewSource(23))
	x := make([][4]float32, n)
	flat := make([]float64, n*4)
	for i := range x {
		for j := 0; j < 4; j++ {
			x[i][j] = float32(r.NormFloat64())
			flat[i*4+j] = float64(x[i][j])
		}
	}
	levels := lloyd(flat, 8, 40)
	var num, den float64
	for _, v := range flat {
		den += v * v
		d := v - nearestLevel(levels, v)
		num += d * d
	}
	scalar := -10 * math.Log10(num/den)

	shell := bestScale(x, func(v, out []float32) {
		nearestDn(v, out)
		for norm2(out) > float32(nn.D4Radius) {
			for j := range v {
				v[j] *= 0.97
			}
			nearestDn(v, out)
		}
	})
	t.Logf("three bits a weight: %.2f dB from D4 in a 16 KiB table, %.2f dB from eight Lloyd levels in registers — the lattice buys %.2f dB",
		shell, scalar, shell-scalar)
}

// lloyd is the Lloyd-Max quantizer: the levels that minimise the squared error
// on a sample, found by alternating between the boundaries the levels imply and
// the centroids the boundaries imply.
func lloyd(x []float64, k, iters int) []float64 {
	levels := make([]float64, k)
	for i := range levels {
		levels[i] = -3 + 6*float64(i)/float64(k-1)
	}
	sums := make([]float64, k)
	counts := make([]int, k)
	for it := 0; it < iters; it++ {
		for i := range sums {
			sums[i], counts[i] = 0, 0
		}
		for _, v := range x {
			b := 0
			for i := 1; i < k; i++ {
				if math.Abs(v-levels[i]) < math.Abs(v-levels[b]) {
					b = i
				}
			}
			sums[b] += v
			counts[b]++
		}
		for i := range levels {
			if counts[i] > 0 {
				levels[i] = sums[i] / float64(counts[i])
			}
		}
	}
	return levels
}

func nearestLevel(levels []float64, v float64) float64 {
	b := 0
	for i := 1; i < len(levels); i++ {
		if math.Abs(v-levels[i]) < math.Abs(v-levels[b]) {
			b = i
		}
	}
	return levels[b]
}

// And the codebook made of shifts.
//
// Representing a weight as a power of two turns its multiplication into a shift,
// which is the whole of logarithmic quantization — INQ, LogNet, ShiftCNN — and
// is worth having on hardware with no multiplier. A GPU has thousands, and this
// kernel is bound by reading weights rather than by arithmetic, so the shift
// buys nothing here. What it costs is another matter, and this is it.
//
// Three bits over a sign and four magnitudes, each half the one before, at
// whichever scale serves it best.
func TestShiftCodebookCosts(t *testing.T) {
	const n = 200000
	r := rand.New(rand.NewSource(29))
	flat := make([]float64, n)
	for i := range flat {
		flat[i] = r.NormFloat64()
	}
	levels := []float64{-1, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1}
	best := 0.0
	for mu := 0.3; mu < 6; mu *= 1.04 {
		var num, den float64
		for _, v := range flat {
			den += v * v
			d := v - nearestLevel(levels, v/mu)*mu
			num += d * d
		}
		if snr := -10 * math.Log10(num/den); snr > best {
			best = snr
		}
	}
	lloydLevels := lloyd(flat, 8, 60)
	var num, den float64
	for _, v := range flat {
		den += v * v
		d := v - nearestLevel(lloydLevels, v)
		num += d * d
	}
	opt := -10 * math.Log10(num/den)
	t.Logf("three bits a weight: %.2f dB from eight powers of two, %.2f dB from Lloyd's eight — shifts cost %.2f dB",
		best, opt, opt-best)
}
