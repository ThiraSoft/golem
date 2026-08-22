package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The ggml softmax against the library's, which is what it stands in for. The
// polynomial is documented at one and a half units in the last place, so the
// bar is a few parts in a million of a probability that sums to one.
func TestSoftmaxGGMLAgreesWithTheLibrary(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for _, n := range []int{1, 2, 7, 8, 9, 64, 1053} {
		x := make([]float32, n)
		for i := range x {
			x[i] = (r.Float32()*2 - 1) * 12
		}
		want := append([]float32(nil), x...)
		SoftmaxInPlace(want)
		got := append([]float32(nil), x...)
		SoftmaxGGML(got)

		var sum float64
		for i := range got {
			sum += float64(got[i])
			if diff := got[i] - want[i]; diff > 2e-6 || diff < -2e-6 {
				t.Fatalf("n=%d element %d is %g, the library gives %g", n, i, got[i], want[i])
			}
		}
		if math.Abs(sum-1) > 1e-4 {
			t.Fatalf("n=%d sums to %g", n, sum)
		}
	}
}

// A row whose spread is wider than the exponential's range: everything far
// below the maximum has to come out as nothing rather than as noise.
func TestSoftmaxGGMLSurvivesAWideRow(t *testing.T) {
	x := []float32{-400, -120, -88, 0, -1e30, 3}
	SoftmaxGGML(x)
	var sum float32
	for i, v := range x {
		if v < 0 || v > 1 || math.IsNaN(float64(v)) {
			t.Fatalf("element %d came out %g", i, v)
		}
		sum += v
	}
	if diff := sum - 1; diff > 1e-5 || diff < -1e-5 {
		t.Fatalf("the row sums to %g", sum)
	}
}
