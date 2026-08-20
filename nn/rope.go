package nn

// The rotation, tabulated.
//
// The angles a position rotates by depend on the position and on the geometry —
// the base, the width, and the frequency factors — and on nothing else. A model
// has two or three geometries and advances one position at a time, so the sines
// and cosines are computed a few hundred times per token here instead of once
// per head per block, which is where the transcendental functions were showing
// up in the profile.

import "math"

// A table rotates one head in place, pairing element i with element i+d/2 —
// the "NeoX" convention, as opposed to the consecutive pairing of ApplyRoPE.
//
// The frequency factors, when present, divide the angle of each pair. They
// number d/2, and a huge one (the conversion writes 1e30) is how a model says a
// pair is not to be rotated at all. Gemma's global blocks arrive that way:
// sixty-four rotated pairs out of two hundred and fifty-six.
type RoPETable struct {
	Cos, Sin []float32

	dims     int
	position int
	base     float64
	factors  []float32
	ready    bool
}

// Prepare makes the table current for this position and geometry, recomputing
// only when one of them has changed.
func (t *RoPETable) Prepare(dims, position int, base float64, factors []float32) {
	if dims%2 != 0 {
		panic("nn: RoPE needs an even head dimension")
	}
	if factors != nil && len(factors) != dims/2 {
		panic("nn: RoPE frequency factors must number half the head dimension")
	}
	if t.ready && t.dims == dims && t.position == position && t.base == base && sameSlice(t.factors, factors) {
		return
	}
	half := dims / 2
	if cap(t.Cos) < half {
		t.Cos = make([]float32, half)
		t.Sin = make([]float32, half)
	}
	t.Cos, t.Sin = t.Cos[:half], t.Sin[:half]
	for i := 0; i < half; i++ {
		theta := float64(position) * math.Pow(base, -2*float64(i)/float64(dims))
		if factors != nil {
			theta /= float64(factors[i])
		}
		t.Cos[i] = float32(math.Cos(theta))
		t.Sin[i] = float32(math.Sin(theta))
	}
	t.dims, t.position, t.base, t.factors, t.ready = dims, position, base, factors, true
}

// Apply rotates one head, which must be as wide as the table was prepared for.
func (t *RoPETable) Apply(vec []float32) {
	half := len(t.Cos)
	if len(vec) != 2*half {
		panic("nn: the rotation table was prepared for another width")
	}
	for i := 0; i < half; i++ {
		cos, sin := t.Cos[i], t.Sin[i]
		re, im := vec[i], vec[i+half]
		vec[i] = re*cos - im*sin
		vec[i+half] = re*sin + im*cos
	}
}

// sameSlice reports whether two slices share their first element, which is how
// the table tells one block's frequency factors from another's.
func sameSlice(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	if len(a) == 0 {
		return true
	}
	return &a[0] == &b[0]
}
