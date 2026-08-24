package nn

import (
	"math"
	"math/rand"
	"testing"
)

// Four columns at once against one at a time.
//
// The lengths matter more than they look. A row of 17 is the shape that broke
// the AVX2 kernel: long enough for the sixteen-at-a-time loop to have run, and
// not a whole number of eights, so the one-at-a-time tail runs after it. Every
// width the engines actually pass here is a multiple of eight, which is why
// nothing noticed.
func TestDotBF16x4AgreesWithOneAtATime(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for _, n := range []int{768, 3072, 16, 8, 24, 13, 1, 4, 7, 12, 15, 17, 25, 33} {
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
			t.Skip("no four-column kernel on this machine")
		}
		for c := 0; c < 4; c++ {
			want := dotBF16(row, x[c*stride:c*stride+n])
			if diff := got[c] - want; diff > 1e-3 || diff < -1e-3 {
				t.Errorf("n=%d column %d: %g against %g", n, c, got[c], want)
			}
		}
	}
}

// Each column reads its own activation, and each weight lands on its own
// element of it.
//
// A one-hot column picks out a single weight and returns it with no rounding,
// so this pins both mappings at once: a kernel that crossed its columns, or
// that walked one of them at the wrong stride, returns a value from somewhere
// else and this says which.
func TestDotBF16x4KeepsItsColumnsApart(t *testing.T) {
	r := rand.New(rand.NewSource(29))
	for _, n := range []int{1, 4, 7, 8, 12, 16, 17, 33} {
		row := make([]uint16, n)
		for i := range row {
			row[i] = uint16(math.Float32bits(r.Float32()*4-2) >> 16)
		}
		const stride = 512
		for hot := 0; hot < n; hot++ {
			x := make([]float32, 4*stride)
			// A different weight for each column, so a crossed pair shows.
			for c := 0; c < 4; c++ {
				x[c*stride+(hot+c)%n] = 1
			}
			var got [4]float32
			if !dotBF16x4(row, x, stride, n, &got) {
				t.Skip("no four-column kernel on this machine")
			}
			for c := 0; c < 4; c++ {
				if want := bf16(row[(hot+c)%n]); got[c] != want {
					t.Fatalf("n=%d, weight %d, column %d: %v against %v", n, hot, c, got[c], want)
				}
			}
		}
	}
}
