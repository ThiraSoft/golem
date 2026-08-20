package nn

import (
	"math"
	"math/rand"
	"testing"
)

func TestDotF32(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for _, n := range []int{0, 1, 5, 8, 9, 31, 32, 33, 256, 1000} {
		a := make([]float32, n)
		b := make([]float32, n)
		var want float64
		for i := range a {
			a[i] = r.Float32()*2 - 1
			b[i] = r.Float32()*2 - 1
			want += float64(a[i]) * float64(b[i])
		}
		got := DotF32(a, b)
		if d := math.Abs(float64(got) - want); d > 1e-4*(1+math.Abs(want)) {
			t.Fatalf("n=%d: %v against %v", n, got, want)
		}
	}
}
