package nn

import (
	"math"
	"math/rand"
	"testing"
)

func TestDotAVX2MatchesScalar(t *testing.T) {
	if !avx2 {
		t.Skip("no AVX2")
	}
	r := rand.New(rand.NewSource(1))
	for _, n := range []int{1, 3, 7, 8, 9, 31, 32, 33, 63, 100, 1024, 4096} {
		w := make([]uint16, n)
		x := make([]float32, n)
		for i := range w {
			w[i] = uint16(math.Float32bits(r.Float32()*2-1) >> 16)
			x[i] = r.Float32()*2 - 1
		}
		var want float32
		for i := range w {
			want += math.Float32frombits(uint32(w[i])<<16) * x[i]
		}
		got := dotBF16AVX2(&w[0], &x[0], n)
		if d := math.Abs(float64(got - want)); d > 1e-4*(1+math.Abs(float64(want))) {
			t.Fatalf("n=%d: %v vs %v", n, got, want)
		}
	}
}
