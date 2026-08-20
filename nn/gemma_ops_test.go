package nn

import (
	"math"
	"testing"
)

func TestRMSNormPlain(t *testing.T) {
	x := []float32{1, -2, 3, -4}
	gain := []float32{2, 2, 2, 2}
	RMSNormPlain(x, gain, 1e-6)

	// mean of squares is (1+4+9+16)/4 = 7.5, so the divisor is sqrt(7.5+1e-6)
	inv := 1 / math.Sqrt(7.5+1e-6)
	want := []float32{
		float32(1 * inv * 2), float32(-2 * inv * 2),
		float32(3 * inv * 2), float32(-4 * inv * 2),
	}
	for i := range x {
		if math.Abs(float64(x[i]-want[i])) > 1e-6 {
			t.Fatalf("element %d: %v, want %v", i, x[i], want[i])
		}
	}
}

func TestRMSNormPlainWithoutGain(t *testing.T) {
	x := []float32{3, 4}
	RMSNormPlain(x, nil, 0)
	// mean of squares is 12.5, so the divisor is sqrt(12.5)
	inv := float32(1 / math.Sqrt(12.5))
	if math.Abs(float64(x[0]-3*inv)) > 1e-6 || math.Abs(float64(x[1]-4*inv)) > 1e-6 {
		t.Fatalf("got %v", x)
	}
}

// The mean is not subtracted. A constant vector normalizes to a constant, which
// is the difference from the LayerNorm-shaped RMSNorm the TTS engine uses.
func TestRMSNormPlainDoesNotCentre(t *testing.T) {
	x := []float32{5, 5, 5, 5}
	RMSNormPlain(x, nil, 0)
	for i, v := range x {
		if math.Abs(float64(v-1)) > 1e-6 {
			t.Fatalf("element %d is %v, want 1 — the mean must not be subtracted", i, v)
		}
	}
}

// rotate is the table applied to one vector, which is how the engine rotates a
// head — prepared once for a position, then applied to every head at it.
func rotate(vec []float32, position int, base float64, factors []float32) {
	var t RoPETable
	t.Prepare(len(vec), position, base, factors)
	t.Apply(vec)
}

func TestApplyRoPENeoXAtZeroIsIdentity(t *testing.T) {
	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	want := append([]float32(nil), vec...)
	rotate(vec, 0, 10000, nil)
	for i := range vec {
		if vec[i] != want[i] {
			t.Fatalf("position 0 rotated element %d: %v, want %v", i, vec[i], want[i])
		}
	}
}

// NeoX pairs element i with element i+d/2, not 2i with 2i+1.
func TestApplyRoPENeoXPairing(t *testing.T) {
	const d = 8
	vec := make([]float32, d)
	vec[0], vec[d/2] = 1, 0 // the first pair, as a unit real
	rotate(vec, 3, 10000, nil)

	angle := 3 * math.Pow(10000, -0.0) // pair 0: exponent -2*0/d = 0
	wantRe := float32(math.Cos(angle))
	wantIm := float32(math.Sin(angle))
	if math.Abs(float64(vec[0]-wantRe)) > 1e-6 || math.Abs(float64(vec[d/2]-wantIm)) > 1e-6 {
		t.Fatalf("pair 0 is (%v, %v), want (%v, %v)", vec[0], vec[d/2], wantRe, wantIm)
	}
	// nothing else moved
	for i := 1; i < d/2; i++ {
		if vec[i] != 0 || vec[i+d/2] != 0 {
			t.Fatalf("pair %d moved without cause", i)
		}
	}
}

// A frequency factor of 1e30 is how the conversion says "leave this pair
// alone": it divides the angle into nothing. Gemma's global blocks rotate 64
// pairs out of 256 that way.
func TestApplyRoPENeoXFrequencyFactors(t *testing.T) {
	const d = 8
	vec := []float32{1, 1, 1, 1, 0, 0, 0, 0}
	before := append([]float32(nil), vec...)
	rotate(vec, 7, 10000, []float32{1, 1e30, 1e30, 1e30})

	if vec[0] == before[0] {
		t.Fatal("pair 0 should have rotated")
	}
	// The angle is divided rather than dropped, so what is left of it is of the
	// order of 1e-27 — the same residue ggml carries, and nothing a float32
	// activation can tell from an untouched value.
	for i := 1; i < d/2; i++ {
		if math.Abs(float64(vec[i]-before[i])) > 1e-20 ||
			math.Abs(float64(vec[i+d/2]-before[i+d/2])) > 1e-20 {
			t.Fatalf("pair %d rotated despite a factor of 1e30", i)
		}
	}
}

func TestSoftcap(t *testing.T) {
	x := []float32{0, 30, -300, 6}
	Softcap(x, 30)
	want := []float32{
		0,
		float32(30 * math.Tanh(1)),
		float32(30 * math.Tanh(-10)),
		float32(30 * math.Tanh(0.2)),
	}
	for i := range x {
		if math.Abs(float64(x[i]-want[i])) > 1e-5 {
			t.Fatalf("element %d: %v, want %v", i, x[i], want[i])
		}
	}
}
