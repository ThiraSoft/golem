package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The shell has to be the shell, and exactly as large as the codes: every point
// of even coordinate sum within a squared norm of forty — 3961 of them, which is
// 24·Σσ_odd(n) by D4's theta series — and then enough of the next norm up to
// fill the twelve bits. A point miscounted is a rate misreported and a code that
// means two things; a code left unnamed is a code the weights paid for.
func TestD4ShellIsTheLattice(t *testing.T) {
	if got := D4Points(); got != 1<<D4Bits {
		t.Fatalf("%d points, want %d", got, 1<<D4Bits)
	}
	full := 0
	for i := 0; i < D4Points(); i++ {
		p := D4Point(uint16(i))
		if p[0]*p[0]+p[1]*p[1]+p[2]*p[2]+p[3]*p[3] <= D4Radius {
			full++
		}
	}
	if full != 3961 {
		t.Fatalf("%d points within norm %d, want the theta series' 3961", full, D4Radius)
	}
	if got := D4Points() * 4 * 2; got > 32<<10 {
		t.Fatalf("decode table of %d bytes, over a workgroup's shared memory", got)
	}
	seen := map[[4]int8]bool{}
	for i := 0; i < D4Points(); i++ {
		p := D4Point(uint16(i))
		if seen[p] {
			t.Fatalf("point %v appears twice", p)
		}
		seen[p] = true
		sum := int(p[0]) + int(p[1]) + int(p[2]) + int(p[3])
		if sum&1 != 0 {
			t.Fatalf("point %v has odd coordinate sum", p)
		}
		n2 := int(p[0])*int(p[0]) + int(p[1])*int(p[1]) + int(p[2])*int(p[2]) + int(p[3])*int(p[3])
		if n2 > d4EdgeNorm {
			t.Fatalf("point %v has squared norm %d, past the shell", p, n2)
		}
		if i > 0 {
			// Canonical order, or an encoder and a decoder built a month apart
			// disagree about what a code means.
			q := D4Point(uint16(i - 1))
			m2 := int(q[0])*int(q[0]) + int(q[1])*int(q[1]) + int(q[2])*int(q[2]) + int(q[3])*int(q[3])
			if m2 > n2 {
				t.Fatalf("point %d has norm %d after one of norm %d", i, n2, m2)
			}
		}
		if !D4Has(p, D4Bits) {
			t.Fatalf("point %v has a code but D4Has says it has none", p)
		}
		if c, ok := D4Code(p); !ok || int(c) != i {
			t.Fatalf("point %v codes back to %d, not %d", p, c, i)
		}
	}
}

// Twelve-bit codes packed two to three bytes, and read back.
func TestD4CodePacking(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	codes := make([]uint16, 16)
	buf := make([]byte, 24)
	for trial := 0; trial < 500; trial++ {
		for i := range codes {
			codes[i] = uint16(r.Intn(D4Points()))
		}
		PutD4Codes(buf, codes)
		for i, want := range codes {
			if got := d4CodeAt(buf, i, D4Bits); got != want {
				t.Fatalf("code %d came back %d, want %d", i, got, want)
			}
		}
	}
}

// A block written by hand and read by the dequantiser: each half of the block
// takes its own step, and the four coordinates of a code land on four
// consecutive weights. The two steps are different on purpose — one step for
// the whole block would pass whichever half the reader took it from.
func TestD4GBlockRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	block := make([]byte, d4BlockBytes)
	lo, hi := D4StepCode(0.0125), D4StepCode(0.4)
	block[0], block[1] = lo, hi
	codes := make([]uint16, 16)
	for i := range codes {
		codes[i] = uint16(r.Intn(D4Points()))
	}
	PutD4Codes(block[2:], codes)

	out := make([]float32, D4Block)
	DequantizeD4G(block, D4Block, out)
	for i, c := range codes {
		s := D4Step(lo)
		if i*4 >= D4SubBlock {
			s = D4Step(hi)
		}
		p := D4Point(c)
		for j := 0; j < 4; j++ {
			want := float32(p[j]) * s
			if got := out[i*4+j]; got != want {
				t.Fatalf("weight %d is %g, want %g", i*4+j, got, want)
			}
		}
	}
}

// The step codes have to name the step they were asked for, to the ratio they
// are spaced by, over the range a weight matrix uses — and the range is the
// other half of the bargain: eight bits buy either precision or reach, and the
// balance chosen here is what stops a coarse step from costing more than the
// granularity it pays for.
func TestD4StepCodeRoundTrip(t *testing.T) {
	for _, v := range []float32{1e-5, 1e-4, 3e-3, 0.0125, 0.1, 0.4} {
		got := D4Step(D4StepCode(v))
		if ratio := float64(got / v); ratio < 0.96 || ratio > 1.045 {
			t.Errorf("step %g came back as %g", v, got)
		}
	}
	if lo, hi := D4Step(0), D4Step(255); lo > 1e-5 || hi < 0.4 {
		t.Errorf("the codes reach %g to %g, which does not cover what weights ask for", lo, hi)
	}
	// Past either end the code saturates rather than wrapping, which is the
	// difference between a block that is a little wrong and one that is noise.
	if c := D4StepCode(1e-12); c != 0 {
		t.Errorf("a step under the range named code %d, not 0", c)
	}
	if c := D4StepCode(1e6); c != 255 {
		t.Errorf("a step over the range named code %d, not 255", c)
	}
}

// The rotation has to be its own inverse up to the sign vector it is folded
// with, or the weights and the activations disagree about which space they are
// in and the product is quietly wrong.
func TestPrepareD4GIsOrthogonal(t *testing.T) {
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

	PrepareD4G(x, pre, group)
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

// More than one block, so the split planes are actually exercised: with a
// single block the two layouts coincide and a reader that still thinks the
// scale sits next to its codes would pass.
func TestD4GPlanesAreSplit(t *testing.T) {
	const n = D4Block * 5
	r := rand.New(rand.NewSource(21))
	row := make([]byte, n/D4Block*d4BlockBytes)
	scales, codes := D4Planes(row, n)
	if len(scales) != 10 || len(codes) != 120 {
		t.Fatalf("planes are %d and %d bytes, want 10 and 120", len(scales), len(codes))
	}
	want := make([]float32, n)
	for b := 0; b < 5; b++ {
		lo := D4StepCode(float32(b+1) * 0.01)
		hi := D4StepCode(float32(b+1) * 0.03)
		scales[b*2], scales[b*2+1] = lo, hi
		cs := make([]uint16, 16)
		for i := range cs {
			cs[i] = uint16(r.Intn(D4Points()))
		}
		PutD4Codes(codes[b*24:], cs)
		for i, c := range cs {
			s := D4Step(lo)
			if i*4 >= D4SubBlock {
				s = D4Step(hi)
			}
			p := D4Point(c)
			for j := 0; j < 4; j++ {
				want[b*D4Block+i*4+j] = float32(p[j]) * s
			}
		}
	}
	got := make([]float32, n)
	DequantizeD4G(row, n, got)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("weight %d is %g, want %g", i, got[i], want[i])
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
	PrepareD4G(x, inv, group)
	UnprepareD4G(x, pre, group)
	for i := range x {
		if d := math.Abs(float64(x[i] - want[i])); d > 1e-5 {
			t.Fatalf("value %d came back as %g, not %g", i, x[i], want[i])
		}
	}
}
