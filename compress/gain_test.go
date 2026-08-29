package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// Rounding onto a lattice is supposed to shrink a row — the reconstruction a
// little shorter than what it stands for — and that part of the error would be
// systematic, so one number a row would take it away. It does not: the gain
// comes out at 1.0000 ± 1.7% and removes half a percent of the error. This is
// here so that nobody adds a per-row gain to the format on the strength of the
// argument, which is sound, without the measurement, which says no.
func TestPerRowGainIsNotWorthCarrying(t *testing.T) {
	const rows, cols, samples = 128, 1024, 256
	rng := rand.New(rand.NewSource(5))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	x := make([][]float32, samples)
	for i := range x {
		r := make([]float32, cols)
		var drift float32
		for j := range r {
			drift = 0.9*drift + float32(rng.NormFloat64())
			r[j] = drift
		}
		x[i] = r
	}
	q := RandomSigns(cols, 5)
	inv := make([]float32, cols)
	for i, v := range q {
		inv[i] = 1 / v
	}
	p := D4Params{Beta: 2, ScaleBlock: 32, HadGroup: 128, SearchScale: true}
	data := EncodeD4G(w, rows, cols, q, p, nil)

	xr := cloneRows(x)
	for _, r := range xr {
		nn.PrepareD4G(r, inv, 128)
	}
	stride := cols / nn.D4Block * 26
	rec := make([]float32, cols)
	var plain, gained, den float64
	var sumG, sumG2 float64
	for r := 0; r < rows; r++ {
		nn.DequantizeD4G(data[r*stride:(r+1)*stride], cols, rec)
		orig := w[r*cols : (r+1)*cols]
		// The gain that best matches the product, in closed form.
		var num, dsq float64
		for i, s := range xr {
			var a, b float64
			for j := range s {
				a += float64(orig[j]) * float64(x[i][j])
				b += float64(rec[j]) * float64(s[j])
			}
			num += a * b
			dsq += b * b
		}
		g := num / dsq
		sumG += g
		sumG2 += g * g
		for i, s := range xr {
			var a, b float64
			for j := range s {
				a += float64(orig[j]) * float64(x[i][j])
				b += float64(rec[j]) * float64(s[j])
			}
			plain += (a - b) * (a - b)
			gained += (a - g*b) * (a - g*b)
			den += a * a
		}
	}
	mean := sumG / rows
	sd := math.Sqrt(sumG2/rows - mean*mean)
	t.Logf("gain %.4f ± %.4f; output error %.4f plain, %.4f with a gain a row (%.1f%% of it)",
		mean, sd, math.Sqrt(plain/den), math.Sqrt(gained/den),
		100*math.Sqrt(gained/plain))
}
