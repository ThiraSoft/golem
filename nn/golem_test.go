package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The step codes have to name the step they were asked for, to the ratio they
// are spaced by, over the range a weight matrix uses — and the range is the
// other half of the bargain: eight bits buy either precision or reach, and the
// balance chosen here is what stops a coarse step from costing more than the
// granularity it pays for.
func TestD4StepCodeRoundTrip(t *testing.T) {
	for _, v := range []float32{1e-5, 1e-4, 3e-3, 0.0125, 0.1, 0.4} {
		got := GolemStep(GolemStepCode(v))
		if ratio := float64(got / v); ratio < 0.96 || ratio > 1.045 {
			t.Errorf("step %g came back as %g", v, got)
		}
	}
	if lo, hi := GolemStep(0), GolemStep(255); lo > 1e-5 || hi < 0.4 {
		t.Errorf("the codes reach %g to %g, which does not cover what weights ask for", lo, hi)
	}
	// Past either end the code saturates rather than wrapping, which is the
	// difference between a block that is a little wrong and one that is noise.
	if c := GolemStepCode(1e-12); c != 0 {
		t.Errorf("a step under the range named code %d, not 0", c)
	}
	if c := GolemStepCode(1e6); c != 255 {
		t.Errorf("a step over the range named code %d, not 255", c)
	}
}

// The rotation has to be its own inverse up to the sign vector it is folded
// with, or the weights and the activations disagree about which space they are
// in and the product is quietly wrong.
func TestPrepareGolemIsOrthogonal(t *testing.T) {
	const n, group = 256, 128
	r := rand.New(rand.NewSource(11))
	x := make([]float32, n)
	pre := make([]float32, n)
	for i := range x {
		x[i] = r.Float32()*2 - 1
		pre[i] = 1
		if r.Intn(2) == 0 {
			pre[i] = -1
		}
	}
	before := make([]float32, n)
	copy(before, x)

	PrepareGolem(x, pre, group)
	// A norm the transform must preserve, group by group.
	for base := 0; base < n; base += group {
		var a, b float64
		for i := base; i < base+group; i++ {
			a += float64(before[i]) * float64(before[i])
			b += float64(x[i]) * float64(x[i])
		}
		if math.Abs(a-b) > 1e-3*a {
			t.Fatalf("group at %d: norm² %g became %g", base, a, b)
		}
	}
}

// The embedding table is stored the way the head wants it, so the input path
// has to undo that. The two have to compose to the identity, and the order of
// the two steps is where that goes wrong: the transform is an involution and
// the scaling is not, so undoing them in the same order undoes neither.
func TestUnprepareInvertsPrepare(t *testing.T) {
	const n, group = 256, 128
	pre := make([]float32, n)
	x := make([]float32, n)
	rng := rand.New(rand.NewSource(11))
	for i := range x {
		x[i] = float32(rng.NormFloat64())
		s := float32(0.3 + rng.Float64())
		if rng.Intn(2) == 0 {
			s = -s
		}
		pre[i] = s
	}
	want := append([]float32(nil), x...)
	// The weights met 1/pre, so the row comes back through pre.
	inv := make([]float32, n)
	for i, v := range pre {
		inv[i] = 1 / v
	}
	PrepareGolem(x, inv, group)
	UnprepareGolem(x, pre, group)
	for i := range x {
		if d := math.Abs(float64(x[i] - want[i])); d > 1e-5 {
			t.Fatalf("value %d came back as %g, not %g", i, x[i], want[i])
		}
	}
}
