package nn

import (
	"math"
	"math/rand"
	"testing"
)

// Four columns at once against one at a time, on the sizes this kernel meets:
// a multiple of sixteen, one of eight, and one of neither.
func TestDotBF16x4AgreesWithOneAtATime(t *testing.T) {
	if !avx2 {
		t.Skip("no AVX2 on this machine")
	}
	r := rand.New(rand.NewSource(7))
	for _, n := range []int{768, 3072, 16, 8, 24, 13, 1} {
		row := make([]uint16, n)
		for i := range row {
			row[i] = uint16(math.Float32bits(r.Float32()*4-2) >> 16)
		}
		const stride = 4096 // wider than n, so the columns do not touch
		x := make([]float32, 4*stride)
		for i := range x {
			x[i] = r.Float32()*2 - 1
		}
		var got [4]float32
		if !dotBF16x4(row, x, stride, n, &got) {
			t.Fatal("the kernel declined")
		}
		for c := 0; c < 4; c++ {
			want := dotBF16(row, x[c*stride:c*stride+n])
			if diff := got[c] - want; diff > 1e-3 || diff < -1e-3 {
				t.Errorf("n=%d column %d: %g against %g", n, c, got[c], want)
			}
		}
	}
}
