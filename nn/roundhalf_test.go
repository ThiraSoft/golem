package nn

import (
	"math"
	"math/rand"
	"testing"
)

// Bit-identical to the scalar rounding, on every length and on the values that
// break naive conversions: subnormals, the edges of the exponent range, and the
// exact halves that decide a tie.
func TestRoundHalfRangeMatchesRoundHalf(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{1, 2, 7, 8, 9, 31, 32, 4000} {
		v := make([]float32, n)
		for i := range v {
			switch i % 4 {
			case 0:
				v[i] = float32(rng.NormFloat64())
			case 1:
				v[i] = float32(rng.NormFloat64()) * 1e-6
			case 2:
				v[i] = float32(rng.NormFloat64()) * 1e4
			default:
				v[i] = float32(rng.Intn(2049)) / 2048
			}
		}
		want := make([]float32, n)
		for i, x := range v {
			want[i] = RoundHalf(x)
		}
		got := append([]float32(nil), v...)
		RoundHalfRange(got)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("n=%d at %d: %v rounded to %v, RoundHalf gives %v",
					n, i, v[i], got[i], want[i])
			}
		}
	}
}

func TestRoundHalfRangeHandlesExtremes(t *testing.T) {
	v := []float32{0, -0, 1, -1, 65504, -65504, 1e-8, -1e-8, 6e-8,
		float32(math.Inf(1)), float32(math.Inf(-1))}
	want := make([]float32, len(v))
	for i, x := range v {
		want[i] = RoundHalf(x)
	}
	got := append([]float32(nil), v...)
	RoundHalfRange(got)
	for i := range got {
		if got[i] != want[i] && !(math.IsNaN(float64(got[i])) && math.IsNaN(float64(want[i]))) {
			t.Fatalf("at %d: %v rounded to %v, RoundHalf gives %v", i, v[i], got[i], want[i])
		}
	}
}
