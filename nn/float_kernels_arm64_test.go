//go:build arm64

package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The float kernels against the portable loops.
//
// These agree to rounding rather than exactly, and unavoidably so: VFMLA fuses
// the multiply and the add, keeping a wider intermediate than a separate
// multiply and add would, so a lane can differ in the last bit from the value
// Go computes. That is the same latitude the AVX2 kernels are held to, and the
// same 1e-3 relative bound their tests use.
//
// Lengths are chosen to walk every path: a whole number of sixteens, of fours,
// and remainders of one, two and three on top of each.
func TestDotF32NEONMatchesTheGoForm(t *testing.T) {
	rng := rand.New(rand.NewSource(43))

	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 18, 19, 31, 32, 33, 64, 128, 129, 1023, 1024} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = rng.Float32()*2 - 1
			b[i] = rng.Float32()*2 - 1
		}

		var want float32
		var s0, s1, s2, s3 float32
		i := 0
		for ; i+3 < n; i += 4 {
			s0 += a[i] * b[i]
			s1 += a[i+1] * b[i+1]
			s2 += a[i+2] * b[i+2]
			s3 += a[i+3] * b[i+3]
		}
		for ; i < n; i++ {
			s0 += a[i] * b[i]
		}
		want = (s0 + s1) + (s2 + s3)

		got := DotF32(a, b)
		if gap := math.Abs(float64(got - want)); gap > 1e-3*math.Abs(float64(want))+1e-6 {
			t.Fatalf("n=%d: NEON %.8f, portable %.8f", n, got, want)
		}
	}
}

func TestAxpyNEONMatchesTheGoForm(t *testing.T) {
	rng := rand.New(rand.NewSource(47))

	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 19, 31, 32, 33, 64, 129, 1024} {
		src := make([]float32, n)
		dst := make([]float32, n)
		want := make([]float32, n)
		for i := range src {
			src[i] = rng.Float32()*2 - 1
			dst[i] = rng.Float32()*2 - 1
			want[i] = dst[i]
		}
		a := rng.Float32()*2 - 1

		for i := 0; i < n; i++ {
			want[i] += a * src[i]
		}
		Axpy(dst, src, a)

		for i := 0; i < n; i++ {
			if gap := math.Abs(float64(dst[i] - want[i])); gap > 1e-3*math.Abs(float64(want[i]))+1e-6 {
				t.Fatalf("n=%d element %d: NEON %.8f, portable %.8f", n, i, dst[i], want[i])
			}
		}
	}
}

// Axpy must leave everything past n alone. A store that ran one vector too far
// would be invisible in the test above and corrupt the caller's next field.
func TestAxpyNEONWritesNoFurtherThanItShould(t *testing.T) {
	rng := rand.New(rand.NewSource(53))

	for _, n := range []int{1, 3, 5, 15, 17, 31, 33, 63, 65} {
		const guard = 8
		dst := make([]float32, n+guard)
		src := make([]float32, n)
		for i := range src {
			src[i] = rng.Float32()
		}
		for i := range dst {
			dst[i] = 1234.5
		}
		Axpy(dst[:n], src, 2)
		for i := n; i < n+guard; i++ {
			if dst[i] != 1234.5 {
				t.Fatalf("n=%d: element %d past the end was written (%v)", n, i, dst[i])
			}
		}
	}
}

// Scores and Mix are the attention loop, and they are what these two kernels
// were written for — so they are checked through it rather than only at the
// primitives.
func TestScoresAndMixOnNEON(t *testing.T) {
	rng := rand.New(rand.NewSource(59))

	for _, hd := range []int{64, 128, 256} {
		for _, n := range []int{1, 3, 17} {
			q := make([]float32, hd)
			k := make([]float32, hd*n)
			v := make([]float32, hd*n)
			for i := range q {
				q[i] = rng.Float32()*2 - 1
			}
			for i := range k {
				k[i] = rng.Float32()*2 - 1
				v[i] = rng.Float32()*2 - 1
			}

			out := make([]float32, n)
			Scores(q, k, hd, n, out)
			for j := 0; j < n; j++ {
				var want float32
				for i := 0; i < hd; i++ {
					want += q[i] * k[j*hd+i]
				}
				if gap := math.Abs(float64(out[j] - want)); gap > 1e-3*math.Abs(float64(want))+1e-5 {
					t.Fatalf("hd=%d n=%d score %d: %v, want %v", hd, n, j, out[j], want)
				}
			}

			w := make([]float32, n)
			for i := range w {
				w[i] = rng.Float32()
			}
			dst := make([]float32, hd)
			want := make([]float32, hd)
			Mix(dst, v, w, hd, n)
			for j := 0; j < n; j++ {
				for i := 0; i < hd; i++ {
					want[i] += w[j] * v[j*hd+i]
				}
			}
			for i := 0; i < hd; i++ {
				if gap := math.Abs(float64(dst[i] - want[i])); gap > 1e-3*math.Abs(float64(want[i]))+1e-5 {
					t.Fatalf("hd=%d n=%d mix %d: %v, want %v", hd, n, i, dst[i], want[i])
				}
			}
		}
	}
}

// AxpyHalf is the other half of the attention loop — Mix, against a cache kept
// in fp16 — and it is called once per position per head, so it gets the same
// scrutiny as the product: values, and the guard past the end.
func TestAxpyHalfNEONMatchesTheGoForm(t *testing.T) {
	rng := rand.New(rand.NewSource(61))

	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 19, 31, 64, 129, 1024} {
		src := make([]uint16, n)
		dst := make([]float32, n)
		want := make([]float32, n)
		for i := range src {
			src[i] = Half(float32(rng.NormFloat64()))
			dst[i] = rng.Float32()*2 - 1
			want[i] = dst[i]
		}
		a := rng.Float32()*2 - 1

		for i := 0; i < n; i++ {
			want[i] += a * Widen(src[i])
		}
		AxpyHalf(dst, src, a)

		for i := 0; i < n; i++ {
			if gap := math.Abs(float64(dst[i] - want[i])); gap > 1e-3*math.Abs(float64(want[i]))+1e-6 {
				t.Fatalf("n=%d element %d: NEON %.8f, portable %.8f", n, i, dst[i], want[i])
			}
		}
	}
}

func TestAxpyHalfNEONWritesNoFurtherThanItShould(t *testing.T) {
	rng := rand.New(rand.NewSource(67))

	for _, n := range []int{1, 3, 5, 7, 15, 17, 31, 33} {
		const guard = 8
		dst := make([]float32, n+guard)
		src := make([]uint16, n)
		for i := range src {
			src[i] = Half(rng.Float32())
		}
		for i := range dst {
			dst[i] = 987.25
		}
		AxpyHalf(dst[:n], src, 2)
		for i := n; i < n+guard; i++ {
			if dst[i] != 987.25 {
				t.Fatalf("n=%d: element %d past the end was written (%v)", n, i, dst[i])
			}
		}
	}
}
