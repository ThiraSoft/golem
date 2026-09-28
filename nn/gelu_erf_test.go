package nn

import (
	"math"
	"testing"
)

// GELUErf against the float64 GELU, kernel lanes and tail alike, from far
// below zero, where it vanishes, to far above, where it is x.
func TestGELUErfMatchesFloat64(t *testing.T) {
	var xs []float32
	for v := -12.0; v <= 12; v += 0.00137 {
		xs = append(xs, float32(v))
	}
	xs = append(xs, 0, -0, 30, -30, 1e-6, -1e-6, 5000)
	got := append([]float32(nil), xs...)
	GELUErf(got)
	for i, x := range xs {
		d := float64(x)
		want := 0.5 * d * (1 + math.Erf(d/math.Sqrt2))
		// Relative to the value where it is above one, absolute below: in
		// the negative tail the exponent is a sum of terms near ten, and
		// float32 carries it to a few parts in ten million of those.
		if e := math.Abs(float64(got[i])-want) / math.Max(math.Abs(want), 1); e > 5e-7 {
			t.Fatalf("GELU(%v) = %v, want %v (relative %.3g)", x, got[i], want, e)
		}
	}
}

// SoftmaxF32 against the float64 softmax, with a masked score among them.
func TestSoftmaxF32MatchesFloat64(t *testing.T) {
	for _, size := range []int{43, 48} {
		checkSoftmaxF32(t, size)
	}
}

func checkSoftmaxF32(t *testing.T, size int) {
	x := make([]float32, size)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*1.7) * 9)
	}
	x[5] = float32(math.Inf(-1))
	want := make([]float64, len(x))
	var sum float64
	for i, v := range x {
		want[i] = math.Exp(float64(v) - 9)
		sum += want[i]
	}
	got := append([]float32(nil), x...)
	SoftmaxF32(got)
	for i := range want {
		want[i] /= sum
		if math.Abs(float64(got[i])-want[i]) > 1e-6*want[i]+1e-12 {
			t.Fatalf("p[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}
