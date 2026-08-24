package nn

import (
	"math"
	"math/rand"
	"testing"
)

// bf16 is the reference widening: a bfloat16 is a float32 with its low mantissa
// cut away, so putting one back is a shift.
func bf16(w uint16) float32 { return math.Float32frombits(uint32(w) << 16) }

// scalarDotBF16 is the loop the kernels replace, written out here so the
// comparison is against something this file owns rather than against whichever
// path dotBF16 happens to take.
func scalarDotBF16(row []uint16, x []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < len(row); i += 4 {
		s0 += bf16(row[i]) * x[i]
		s1 += bf16(row[i+1]) * x[i+1]
		s2 += bf16(row[i+2]) * x[i+2]
		s3 += bf16(row[i+3]) * x[i+3]
	}
	for ; i < len(row); i++ {
		s0 += bf16(row[i]) * x[i]
	}
	return s0 + s1 + s2 + s3
}

// Whatever kernel this machine has agrees with that loop.
//
// The lengths cross every boundary the kernels have: the block of sixteen, the
// group of four under it, and the one-at-a-time tail under that.
func TestDotBF16MatchesTheScalarLoop(t *testing.T) {
	r := rand.New(rand.NewSource(19))
	for n := 1; n <= 40; n++ {
		runDotBF16Case(t, r, n)
	}
	for _, n := range []int{63, 64, 65, 127, 128, 129, 768, 1024, 3072} {
		runDotBF16Case(t, r, n)
	}
}

func runDotBF16Case(t *testing.T, r *rand.Rand, n int) {
	t.Helper()
	row := make([]uint16, n)
	x := make([]float32, n)
	for i := range row {
		row[i] = uint16(math.Float32bits(r.Float32()*4-2) >> 16)
		x[i] = r.Float32()*2 - 1
	}
	got, want := dotBF16(row, x), scalarDotBF16(row, x)
	if d := math.Abs(float64(got - want)); d > 1e-4*(1+math.Abs(float64(want))) {
		t.Errorf("n=%d: %v against %v", n, got, want)
	}
}

// Every weight reaches the arithmetic in its own position, and widened exactly.
//
// A dot product against a one-hot activation is a single weight, and it comes
// back with no rounding at all: 1.0 times a widened bfloat16 is that float32.
// So this pins the element-to-lane mapping to the bit — a kernel that
// interleaved its halves the other way round, or crossed the low four of a
// vector with the high four, returns a neighbour's weight and this says so.
func TestDotBF16PutsEveryWeightInItsOwnPlace(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	for _, n := range []int{1, 4, 7, 8, 16, 17, 33} {
		row := make([]uint16, n)
		for i := range row {
			row[i] = uint16(math.Float32bits(r.Float32()*4-2) >> 16)
		}
		for hot := 0; hot < n; hot++ {
			x := make([]float32, n)
			x[hot] = 1
			if got, want := dotBF16(row, x), bf16(row[hot]); got != want {
				t.Fatalf("n=%d, weight %d: %v against %v", n, hot, got, want)
			}
		}
	}
}
