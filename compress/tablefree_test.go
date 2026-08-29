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
