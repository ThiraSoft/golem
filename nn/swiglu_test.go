package nn

import (
	"math"
	"testing"
)

// The polynomial against the exact exponential, over the range a feed forward
// actually produces. ggml documents its routine at 1.45358 plus half a unit in
// the last place, so two units is the bar: loose enough to be a fact about the
// routine, tight enough to catch a transcription error in one of its nine
// constants.
//
// Both sides take the same float32 argument. Comparing against math.Exp of the
// float64 the loop counts with would measure the cast instead: exp amplifies a
// difference in its argument by the size of the argument, so at x = 40 the
// rounding of x alone is twenty units in the last place.
func TestExpfMatchesTheExactExponential(t *testing.T) {
	worst, at := 0.0, float32(0)
	for x := -40.0; x <= 40.0; x += 0.0005 {
		xf := float32(x)
		exact := float32(math.Exp(float64(xf)))
		if exact == 0 || math.IsInf(float64(exact), 0) {
			continue
		}
		ulp := math.Abs(float64(expf(xf)-exact)) / math.Abs(float64(exact)) / 1.1920929e-7
		if ulp > worst {
			worst, at = ulp, xf
		}
	}
	if worst > 2 {
		t.Fatalf("worst %.2f units in the last place, at x = %g", worst, at)
	}
	t.Logf("worst %.2f units in the last place, at x = %g", worst, at)
}

// The ends, where the fast path hands over to the exact routine. The answers
// are float32 answers: exp(100) is an infinity there, and exp(-200) a zero,
// whatever float64 would have said.
func TestExpfAtTheExtremes(t *testing.T) {
	for _, x := range []float32{-200, -120, -100, -88, -50, 0, 50, 88, 100, 120, 200} {
		got := expf(x)
		want := float32(math.Exp(float64(x)))
		switch {
		case math.IsInf(float64(want), 1):
			if !math.IsInf(float64(got), 1) {
				t.Errorf("expf(%g) = %g, want an infinity", x, got)
			}
		case want == 0:
			if got != 0 {
				t.Errorf("expf(%g) = %g, want zero", x, got)
			}
		default:
			// Denormals carry fewer bits, so the bar widens where they start.
			bar := 1e-6
			if math.Abs(float64(want)) < 1.2e-38 {
				bar = 1e-1
			}
			if d := math.Abs(float64(got-want)) / math.Abs(float64(want)); d > bar {
				t.Errorf("expf(%g) = %g, want %g", x, got, want)
			}
		}
	}
	if got := expf(0); got != 1 {
		t.Errorf("expf(0) = %g, want exactly 1", got)
	}
}

// SwiGLURange is silu(gate)*up, and it writes into gate.
func TestSwiGLURange(t *testing.T) {
	gate := []float32{-3, -1, 0, 1, 3, 100}
	up := []float32{2, 2, 2, 2, 2, 2}
	want := make([]float32, len(gate))
	for i, g := range gate {
		want[i] = float32(float64(g)/(1+math.Exp(float64(-g)))) * up[i]
	}

	SwiGLURange(gate, up, 0, len(gate))
	for i := range gate {
		if want[i] == 0 {
			if gate[i] != 0 {
				t.Errorf("element %d: %g, want 0", i, gate[i])
			}
			continue
		}
		if d := math.Abs(float64(gate[i]-want[i])) / math.Abs(float64(want[i])); d > 1e-6 {
			t.Errorf("element %d: %g, want %g", i, gate[i], want[i])
		}
	}
}

// It must touch only the stretch it is given: the section that calls it owns
// one range of a shared buffer.
func TestSwiGLURangeStaysInItsRange(t *testing.T) {
	gate := []float32{1, 1, 1, 1}
	up := []float32{5, 5, 5, 5}
	SwiGLURange(gate, up, 1, 3)
	if gate[0] != 1 || gate[3] != 1 {
		t.Errorf("the ends were written: %v", gate)
	}
	if gate[1] == 1 || gate[2] == 1 {
		t.Errorf("the middle was not written: %v", gate)
	}
}

// Both benchmarks copy a pristine gate in before each call. Without that the
// operation feeds on its own output: silu(x)*1.5 applied a thousand times over
// walks the values into the denormals, where every exponential takes the slow
// path and the figure measures that instead.
func benchSource(n int) (src, up []float32) {
	src = make([]float32, n)
	up = make([]float32, n)
	for i := range src {
		src[i] = float32(i%64) - 32
		up[i] = 1.5
	}
	return src, up
}

func BenchmarkSwiGLURange(b *testing.B) {
	const n = 3072
	src, up := benchSource(n)
	gate := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(gate, src)
		SwiGLURange(gate, up, 0, n)
	}
}

// The same shape through the exact exponential and a separate multiply, which
// is what this replaced.
func BenchmarkSiLUExactThenMultiply(b *testing.B) {
	const n = 3072
	src, up := benchSource(n)
	gate := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(gate, src)
		SiLU(gate)
		for j := range gate {
			gate[j] *= up[j]
		}
	}
}

// The copy alone, so the two figures above can be read net of it.
func BenchmarkBenchSourceCopy(b *testing.B) {
	const n = 3072
	src, _ := benchSource(n)
	gate := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(gate, src)
	}
}

func TestGEGLUQuickRange(t *testing.T) {
	gate := []float32{0, 1, -1, 2.5}
	up := []float32{1, 2, 3, 4}
	want := make([]float32, len(gate))
	for i, g := range gate {
		want[i] = float32(float64(g)/(1+math.Exp(-1.702*float64(g))) * float64(up[i]))
	}
	GEGLUQuickRange(gate, up, 0, len(gate))
	for i := range want {
		if diff := gate[i] - want[i]; diff > 1e-5 || diff < -1e-5 {
			t.Errorf("element %d is %g, expected %g", i, gate[i], want[i])
		}
	}
}
