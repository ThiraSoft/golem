package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// Rounding onto a trellis path is supposed to shrink a row — the
// reconstruction a little shorter than what it stands for — and that part of
// the error would be systematic, so one number a row would take it away. It
// does not: the gain comes out close to 1.0000 and removes a fraction of a
// percent of the error. This is here so that nobody adds a per-row gain to the
// format on the strength of the argument, which is sound, without the
// measurement, which says no. TrellisOpts already carries a Gain field for the
// codebook itself; this is the separate question of a gain per row of output.
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
	p := GolemParams{ScaleBlock: nn.T4GBlock, HadGroup: 128}
	data := EncodeT4GAs(w, rows, cols, q, p, nn.T4G)

	xr := cloneRows(x)
	for _, r := range xr {
		nn.PrepareGolem(r, inv, 128)
	}
	m := nn.Matrix{Data: data, Quant: nn.T4G, Rows: rows, Cols: cols}
	rec := make([]float32, cols)
	var plain, gained, den float64
	var sumG, sumG2 float64
	for r := 0; r < rows; r++ {
		m.Row(r, rec)
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

// cloneRows makes an independent copy of each row, for a caller that is about
// to transform one copy in place and still wants the original beside it.
func cloneRows(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, r := range x {
		out[i] = append([]float32(nil), r...)
	}
	return out
}
