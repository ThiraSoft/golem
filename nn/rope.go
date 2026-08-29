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
	at       [4]int
	sections Sections
	base     float64
	factors  []float32
	ready    bool
}

// Sections are the four M-RoPE section widths a file declares, in the order
// ggml reads them: time, height, width, and the extra one a vision encoder
// uses. All zero means a model whose positions are plain integers.
type Sections [4]int

// Prepare makes the table current for a scalar position, which is a position
// whose four components agree.
func (t *RoPETable) Prepare(dims, position int, base float64, factors []float32) {
	t.PrepareMulti(dims, [4]int{position, position, position, position}, base, Sections{}, factors)
}

// PrepareMulti makes the table current for a position that has four components
// and the geometry that says which of them each pair turns by.
//
// The selection is ggml_mrope_cache_init's, under GGML_ROPE_TYPE_IMROPE: the
// pairs take the components in turn — t, h, w, t, h, w — each section stopping
// once it has had its share. What no section changes is the frequency: pair i
// turns at base^(-2i/dims) whichever component it reads, because ggml advances
// all four thetas on every pair and never resets one. That is what makes a
// position whose components agree give exactly what Prepare gives, rather than
// nearly.
func (t *RoPETable) PrepareMulti(dims int, at [4]int, base float64, sections Sections, factors []float32) {
	if dims%2 != 0 {
		panic("nn: RoPE needs an even head dimension")
	}
	if factors != nil && len(factors) != dims/2 {
		panic("nn: RoPE frequency factors must number half the head dimension")
	}
	sect := sections[0] + sections[1] + sections[2] + sections[3]
	if sect > dims/2 {
		panic("nn: the M-RoPE sections are wider than the head")
	}
	if t.ready && t.dims == dims && t.at == at && t.sections == sections &&
		t.base == base && sameSlice(t.factors, factors) {
		return
	}
	half := dims / 2
	if cap(t.Cos) < half {
		t.Cos = make([]float32, half)
		t.Sin = make([]float32, half)
	}
	t.Cos, t.Sin = t.Cos[:half], t.Sin[:half]
	for i := 0; i < half; i++ {
		theta := float64(at[section(i, sect, sections)]) * math.Pow(base, -2*float64(i)/float64(dims))
		if factors != nil {
			theta /= float64(factors[i])
		}
		t.Cos[i] = float32(math.Cos(theta))
		t.Sin[i] = float32(math.Sin(theta))
	}
	t.dims, t.at, t.sections, t.base, t.factors, t.ready = dims, at, sections, base, factors, true
}

// PrepareVision is the rotation a vision tower turns its patches by:
// GGML_ROPE_TYPE_VISION, where the sections are contiguous and each starts its
// frequency again from the top.
//
// That last part is the whole difference from PrepareMulti, and it is not a
// flag. There, pair i turns at base^(-2i/dims) whichever axis it reads, which
// is what makes equal axes collapse into the scalar rotation. Here the ladder
// restarts, so the first pair of the second section turns as fast as the first
// pair of the first. Two rules, two functions.
//
// at is the patch's row and column. VISION ignores the last two sections, so a
// tower declares four and only the first two are ever selected.
//
// head is the whole head, not the n_dims a graph passes to ggml_rope_multi.
// For VISION ggml rotates ne0 — every element — and uses n_dims as the offset
// that pairs element i with element i+n_dims, which is half the head. So a
// head of 72 gives 36 pairs covering all of it, and the frequency ladder is
// taken over 72. Reading n_dims as a count instead builds half the table and
// leaves the second section unrotated, which is right at the origin and wrong
// everywhere else.
func (t *RoPETable) PrepareVision(head int, at [2]int, base float64, sections Sections, factors []float32) {
	dims := head
	if dims%2 != 0 {
		panic("nn: RoPE needs an even head dimension")
	}
	if factors != nil && len(factors) != dims/2 {
		panic("nn: RoPE frequency factors must number half the head dimension")
	}
	sect := sections[0] + sections[1] + sections[2] + sections[3]
	if sect == 0 {
		panic("nn: the vision rotation needs sections")
	}
	// The two rules share a table and therefore a cache key. The negative
	// marker is what keeps a vision table from being handed back for a
	// PrepareMulti with the same numbers: no position is negative.
	key := [4]int{at[0], at[1], -1, -1}
	if t.ready && t.dims == dims && t.at == key && t.sections == sections &&
		t.base == base && sameSlice(t.factors, factors) {
		return
	}
	half := dims / 2
	if cap(t.Cos) < half {
		t.Cos = make([]float32, half)
		t.Sin = make([]float32, half)
	}
	t.Cos, t.Sin = t.Cos[:half], t.Sin[:half]
	for i := 0; i < half; i++ {
		axis, within := visionSection(i%sect, sections)
		// within, not i: this is the reset, and it is the whole of the rule.
		//
		// The denominator is half the head, not the head. ggml takes
		// theta_scale as freq_base^(-2/n_dims), and for VISION ggml.h requires
		// n_dims to be head_size/2 — so the ladder is steeper than the number
		// of pairs would suggest, and the two numbers being equal here is a
		// constraint of the type rather than a coincidence.
		theta := float64(at[axis]) * math.Pow(base, -2*float64(within)/float64(half))
		if factors != nil {
			theta /= float64(factors[i])
		}
		t.Cos[i] = float32(math.Cos(theta))
		t.Sin[i] = float32(math.Sin(theta))
	}
	t.dims, t.at, t.sections, t.base, t.factors, t.ready = dims, key, sections, base, factors, true
}

// visionSection is which axis a sector reads and how far into that section it
// sits, which is the index its frequency is taken from. VISION ignores the
// last two sections, so anything past the second reads the second.
func visionSection(sector int, sections Sections) (axis, within int) {
	if sector < sections[0] {
		return 0, sector
	}
	return 1, sector - sections[0]
}

// section is which of the four components pair i turns by. A model with no
// sections turns everything by the first, which is the scalar rotation.
func section(i, sect int, sections Sections) int {
	if sect == 0 {
		return 0
	}
	sector := i % sect
	switch {
	case sector%3 == 1 && sector < 3*sections[1]:
		return 1
	case sector%3 == 2 && sector < 3*sections[2]:
		return 2
	case sector%3 == 0 && sector < 3*sections[0]:
		return 0
	default:
		return 3
	}
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
