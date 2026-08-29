package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The codes have to survive the packing: sixty-four of three bits into
// twenty-four bytes, five of every eight straddling two of them. A packer that
// dropped the straddle would still round-trip the codes that do not.
func TestL8CodePacking(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	codes := make([]byte, L8Block)
	for i := range codes {
		codes[i] = byte(r.Intn(8))
	}
	buf := make([]byte, 24)
	PutL8Codes(buf, codes)
	for i, want := range codes {
		if got := l8CodeAt(buf, i); got != want {
			t.Fatalf("code %d came back as %d, want %d", i, got, want)
		}
	}
}

// The levels have to be the ones the codebook claims: symmetric, ordered, and
// the boundaries between them the midpoints, so that L8Code names the nearest
// and not merely a near one.
func TestL8LevelsAreLloyd(t *testing.T) {
	for i := 0; i < 4; i++ {
		if a, b := l8Levels[i], -l8Levels[7-i]; math.Abs(float64(a-b)) > 1e-6 {
			t.Errorf("level %d is %g and its mirror %g", i, a, b)
		}
		if l8Levels[i] >= l8Levels[i+1] {
			t.Errorf("the levels are not ordered at %d", i)
		}
	}
	// Every level names itself, and a value just either side of a boundary
	// names the level that side.
	for i, v := range l8Levels {
		if c := L8Code(v); int(c) != i {
			t.Errorf("level %d (%g) names code %d", i, v, c)
		}
	}
	for i, b := range l8Bounds {
		if c := L8Code(b - 1e-4); int(c) != i {
			t.Errorf("just under boundary %d names code %d", i, c)
		}
		if c := L8Code(b + 1e-4); int(c) != i+1 {
			t.Errorf("just over boundary %d names code %d", i, c)
		}
	}
}

// A block written by hand and read back: each half of it under its own step,
// and the levels where the codes say.
func TestL8GBlockRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	const n = L8Block * 3
	row := make([]byte, n/L8Block*l8BlockBytes)
	steps, codes := L8Planes(row, n)
	if len(steps) != 6 || len(codes) != 72 {
		t.Fatalf("planes are %d and %d bytes, want 6 and 72", len(steps), len(codes))
	}
	want := make([]float32, n)
	for b := 0; b < 3; b++ {
		lo := D4StepCode(float32(b+1) * 0.01)
		hi := D4StepCode(float32(b+1) * 0.04)
		steps[b*2], steps[b*2+1] = lo, hi
		cs := make([]byte, L8Block)
		for i := range cs {
			cs[i] = byte(r.Intn(8))
		}
		PutL8Codes(codes[b*24:], cs)
		for i, c := range cs {
			s := D4Step(lo)
			if i >= D4SubBlock {
				s = D4Step(hi)
			}
			want[b*L8Block+i] = l8Levels[c] * s
		}
	}
	got := make([]float32, n)
	DequantizeL8G(row, n, got)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("weight %d is %g, want %g", i, got[i], want[i])
		}
	}
}
