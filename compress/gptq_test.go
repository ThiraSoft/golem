package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// The factorisation is the whole claim: H⁻¹ = RᵀR with R upper triangular. If
// that does not hold the sweep is subtracting noise.
func TestCompFactorInvertsTheHessian(t *testing.T) {
	const n, samples = 32, 128
	rng := rand.New(rand.NewSource(3))
	x := make([][]float32, samples)
	for i := range x {
		row := make([]float32, n)
		var drift float32
		for j := range row {
			drift = 0.7*drift + float32(rng.NormFloat64())
			row[j] = drift
		}
		x[i] = row
	}
	c := NewComp(x, n, n, 0.01)
	r, m := c.block(0)
	if m != n {
		t.Fatalf("window is %d, not %d", m, n)
	}
	// H as the factor saw it, ridge included.
	h := make([]float64, n*n)
	var mean float64
	for _, row := range x {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				h[i*n+j] += float64(row[i]) * float64(row[j])
			}
		}
	}
	for i := 0; i < n; i++ {
		mean += h[i*n+i]
	}
	mean /= n
	for i := 0; i < n; i++ {
		h[i*n+i] += 0.01 * mean
	}
	// (RᵀR)·H must be the identity.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var v float64
			for k := 0; k < n; k++ {
				var inv float64
				for l := 0; l <= k && l <= i; l++ {
					inv += float64(r[l*n+i]) * float64(r[l*n+k])
				}
				v += inv * h[k*n+j]
			}
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(v-want) > 2e-3 {
				t.Fatalf("(R'R H)[%d,%d] = %g, not %g", i, j, v, want)
			}
		}
	}
}

// And the point of it: against activations that are correlated the way real
// ones are, spending the later columns on the error of the earlier ones has to
// leave the product closer than rounding each column on its own.
func TestCompensationLowersTheOutputError(t *testing.T) {
	const rows, cols, samples = 64, 512, 512
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	x := make([][]float32, samples)
	for i := range x {
		row := make([]float32, cols)
		var drift float32
		for j := range row {
			drift = 0.9*drift + float32(rng.NormFloat64())
			row[j] = drift
		}
		x[i] = row
	}
	p := D4Params{Beta: 2, ScaleBlock: 64, SearchScale: true}
	plain := outputError(t, w, rows, cols, x, EncodeD4G(w, rows, cols, nil, p, nil))
	comp := NewComp(cloneRows(x), cols, 256, 0.01)
	fixed := outputError(t, w, rows, cols, x, EncodeD4G(w, rows, cols, nil, p, comp))
	t.Logf("relative output error: %.4f plain, %.4f compensated", plain, fixed)
	if fixed >= plain {
		t.Errorf("compensation left the product no closer: %.4f against %.4f", fixed, plain)
	}
}

func cloneRows(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, r := range x {
		out[i] = append([]float32(nil), r...)
	}
	return out
}

// outputError is ||(W-Ŵ)X|| over ||WX||, which is what the next layer sees.
func outputError(t *testing.T, w []float32, rows, cols int, x [][]float32, data []byte) float64 {
	t.Helper()
	row := make([]float32, cols)
	var num, den float64
	stride := cols / nn.D4Block * 26
	for r := 0; r < rows; r++ {
		nn.DequantizeD4G(data[r*stride:(r+1)*stride], cols, row)
		orig := w[r*cols : (r+1)*cols]
		for _, s := range x {
			var a, b float64
			for j := range s {
				a += float64(orig[j]) * float64(s[j])
				b += float64(row[j]) * float64(s[j])
			}
			num += (a - b) * (a - b)
			den += a * a
		}
	}
	return math.Sqrt(num / den)
}

// Whether the rotation leaves the compensation anything to do. Incoherence
// processing spreads the Hessian towards a multiple of the identity, and a
// diagonal Hessian is exactly the case where there is no error to pass on: the
// columns no longer overlap. The number this prints is the whole reason the
// window is worth what it costs, or is not.
func TestCompensationUnderRotation(t *testing.T) {
	const rows, cols, samples = 64, 512, 1024
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	x := make([][]float32, samples)
	for i := range x {
		row := make([]float32, cols)
		var drift float32
		for j := range row {
			drift = 0.9*drift + float32(rng.NormFloat64())
			row[j] = drift
		}
		x[i] = row
	}
	for _, group := range []int{0, 128} {
		q := RandomSigns(cols, 99)
		inv := make([]float32, cols)
		for i, v := range q {
			inv[i] = 1 / v
		}
		wq := append([]float32(nil), w...)
		xq := cloneRows(x)
		if group == 0 {
			// Unrotated: the weights go in as they are and the activations too.
			copy(wq, w)
			for i := range xq {
				copy(xq[i], x[i])
			}
		}
		g := group
		xr := cloneRows(xq)
		for _, r := range xr {
			nn.PrepareD4G(r, inv, g)
		}
		p := D4Params{Beta: 2, ScaleBlock: 64, HadGroup: g, SearchScale: true}
		var qq []float32
		if g > 0 {
			qq = q
		}
		plain := rotatedError(w, rows, cols, x, EncodeD4G(w, rows, cols, qq, p, nil), qq, g)
		comp := NewComp(xr, cols, 256, 0.01)
		fixed := rotatedError(w, rows, cols, x, EncodeD4G(w, rows, cols, qq, p, comp), qq, g)
		t.Logf("rotation %3d: output error %.4f plain, %.4f compensated (%.0f%% of it left)",
			g, plain, fixed, 100*fixed/plain)
		_ = wq
	}
}

// rotatedError measures the error of the product the way inference computes it:
// the stored row is in the rotated basis, so the activation is put through the
// same transform before the dot.
func rotatedError(w []float32, rows, cols int, x [][]float32, data []byte, q []float32, group int) float64 {
	row := make([]float32, cols)
	stride := cols / nn.D4Block * 26
	var inv []float32
	if q != nil {
		inv = make([]float32, cols)
		for i, v := range q {
			inv[i] = 1 / v
		}
	}
	xs := cloneRows(x)
	if inv != nil {
		for _, r := range xs {
			nn.PrepareD4G(r, inv, group)
		}
	}
	var num, den float64
	for r := 0; r < rows; r++ {
		nn.DequantizeD4G(data[r*stride:(r+1)*stride], cols, row)
		orig := w[r*cols : (r+1)*cols]
		for i, s := range xs {
			var a, b float64
			for j := range s {
				a += float64(orig[j]) * float64(x[i][j])
				b += float64(row[j]) * float64(s[j])
			}
			num += (a - b) * (a - b)
			den += a * a
		}
	}
	return math.Sqrt(num / den)
}
