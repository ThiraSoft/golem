package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The fp16 kernels have to agree with the float32 ones on values that are
// already fp16, which is what the cache holds: it rounds on the way in, so a
// product against it must give what the widened float32 would have given.
func TestDotF32HalfMatchesTheWidenedProduct(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{1, 3, 8, 63, 128, 256} {
		a := make([]float32, n)
		h := make([]uint16, n)
		wide := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			h[i] = Half(float32(rng.NormFloat64()))
			wide[i] = Widen(h[i])
		}
		got, want := DotF32Half(a, h), DotF32(a, wide)
		if d := math.Abs(float64(got - want)); d > 1e-4*math.Abs(float64(want))+1e-6 {
			t.Fatalf("n=%d: fp16 product %v, widened %v", n, got, want)
		}
	}
}

func TestAxpyHalfMatchesTheWidenedSum(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{1, 5, 32, 129} {
		h := make([]uint16, n)
		wide := make([]float32, n)
		for i := range h {
			h[i] = Half(float32(rng.NormFloat64()))
			wide[i] = Widen(h[i])
		}
		// Not a zero start: adding a*v to zero rounds the same whether the
		// multiply and the add are fused or not, so a zero dst hides exactly
		// the difference this test is for.
		got := make([]float32, n)
		want := make([]float32, n)
		for i := range got {
			got[i] = float32(rng.NormFloat64())
			want[i] = got[i]
		}
		AxpyHalf(got, h, 0.75)
		Axpy(want, wide, 0.75)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("n=%d at %d: %v against %v", n, i, got[i], want[i])
			}
		}
	}
}

// Half and Widen are the two ends of what the cache stores, and RoundHalf —
// which the cache used before it held halves — is their composition.
func TestHalfAndWidenAreRoundHalf(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 100000; i++ {
		v := float32(rng.NormFloat64() * 8)
		if got, want := Widen(Half(v)), RoundHalf(v); got != want {
			t.Fatalf("%v widened to %v, RoundHalf gives %v", v, got, want)
		}
	}
}

// The fp16 product must be bit-for-bit what the float32 one gives on the same
// values widened. Anything less and a mixture of experts routes differently:
// its top-k over the router's logits is a hard threshold, and a difference in
// the last bit picks another expert.
func TestDotF32HalfIsBitIdenticalToDotF32(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{1, 2, 7, 8, 9, 31, 32, 33, 64, 128, 255, 256} {
		a := make([]float32, n)
		h := make([]uint16, n)
		wide := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
			h[i] = Half(float32(rng.NormFloat64()))
			wide[i] = Widen(h[i])
		}
		if got, want := DotF32Half(a, h), DotF32(a, wide); got != want {
			t.Fatalf("n=%d: %v against %v, a difference of %v", n, got, want, got-want)
		}
	}
}
