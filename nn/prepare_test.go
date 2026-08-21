package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The prepared activation against the three steps it stands for.
func TestPrepareIlvAgreesWithTheSteps(t *testing.T) {
	r := rand.New(rand.NewSource(41))
	for _, n := range []int{16, 32, 768, 3072} {
		src := make([]float32, n)
		for i := range src {
			src[i] = (r.Float32()*2 - 1) * 8
		}
		const lo, hi = -6.375, 6.3125
		got := make([]float32, n)
		if !PrepareIlv(got, src, lo, hi) {
			t.Skip("no kernel on this machine")
		}
		for g := 0; g < n; g += 16 {
			for i, from := range InterleaveOrder {
				v := src[g+from]
				if v < lo {
					v = lo
				}
				if v > hi {
					v = hi
				}
				want := RoundBF16(v)
				if got[g+i] != want {
					t.Fatalf("n=%d element %d is %v, expected %v",
						n, g+i, got[g+i], want)
				}
			}
		}
	}
}

// The rounding itself, on the values that decide a tie.
func TestPrepareIlvRoundsLikeRoundBF16(t *testing.T) {
	src := make([]float32, 16)
	for i := range src {
		src[i] = math.Float32frombits(uint32(0x3f800000 + i*0x4000))
	}
	got := make([]float32, 16)
	if !PrepareIlv(got, src, -100, 100) {
		t.Skip("no kernel on this machine")
	}
	for i, from := range InterleaveOrder {
		if want := RoundBF16(src[from]); got[i] != want {
			t.Fatalf("element %d is %v (%#x), expected %v (%#x)",
				i, got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
		}
	}
}

func TestClamp(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 768} {
		x := make([]float32, n)
		want := make([]float32, n)
		for i := range x {
			x[i] = float32(i%23) - 11
			want[i] = x[i]
			if want[i] < -3 {
				want[i] = -3
			}
			if want[i] > 4 {
				want[i] = 4
			}
		}
		Clamp(x, -3, 4)
		for i := range want {
			if x[i] != want[i] {
				t.Fatalf("n=%d element %d is %v, expected %v", n, i, x[i], want[i])
			}
		}
	}
}
