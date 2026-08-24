package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The batch product is the vector product repeated, whatever path it takes.
//
// Three of them exist now — the AVX2 branch with its pairs of rows, the
// panelled branch a machine with only the four-column kernel takes, and the
// portable loop — and they are chosen by build tag and by what the processor
// reports, so no single run sees more than one. What holds them together is
// this: each has to give what taking the columns one at a time gives.
//
// The tolerance is relative and loose on purpose. Blocking changes the order
// the sums are taken in, which is the whole point of it; what may not change is
// the answer, to the precision float32 has. The engines that care about the
// last bits of this kernel say so with their own parity fixtures.
func TestMatMatBF16IsTheVectorProductRepeated(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	cases := []struct{ outputs, inputs, batch int }{
		{7, 8, 1},
		{7, 8, 2},
		{7, 8, 4},
		{7, 8, 5},
		{5, 24, 7},
		{16, 64, 4},
		{3, 16, 33}, // more columns than a group of four divides
		{9, 8, 40},
		{4, 1024, 6}, // a wide input, so panelOf falls to its floor
		{6, 40, 37},
	}
	for _, c := range cases {
		w := make([]byte, c.outputs*c.inputs*2)
		words := unsafeWords(w)
		for i := range words {
			words[i] = uint16(math.Float32bits(rng.Float32()*4-2) >> 16)
		}
		x := make([]float32, c.batch*c.inputs)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}

		got := make([]float32, c.batch*c.outputs)
		MatMatBF16(w, x, c.outputs, c.inputs, c.batch, got)

		want := make([]float32, c.batch*c.outputs)
		for l := 0; l < c.batch; l++ {
			MatVecBF16(w, x[l*c.inputs:(l+1)*c.inputs], c.outputs, c.inputs,
				want[l*c.outputs:(l+1)*c.outputs])
		}

		for i := range got {
			d := math.Abs(float64(got[i] - want[i]))
			if d > 1e-4*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("%dx%d batch %d, at %d: %v against %v",
					c.outputs, c.inputs, c.batch, i, got[i], want[i])
			}
		}
	}
}

// A caller that asks for a band of rows writes those rows and no others.
//
// The engines rely on it: they call ApplyRows from inside a parallel section,
// each core owning a band, and a kernel that wrote one row past its share would
// be a race that only shows up under load.
//
// The values in the band are checked loosely, and that is not laziness. Where a
// band starts decides whether a given row is taken by the pair-of-rows kernel
// or the single-row one, so the same row computed under two different splits
// agrees to float32 precision and not to the bit. What must be exact is the
// part this test is really about: everything outside the band still holds the
// value it was given.
func TestMatMatBF16RowsWritesOnlyItsBand(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	const outputs, inputs, batch = 12, 24, 9
	w := make([]byte, outputs*inputs*2)
	words := unsafeWords(w)
	for i := range words {
		words[i] = uint16(math.Float32bits(rng.Float32()*4-2) >> 16)
	}
	x := make([]float32, batch*inputs)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	full := make([]float32, batch*outputs)
	MatMatBF16(w, x, outputs, inputs, batch, full)

	const guard = -12345
	const from, upto = 5, 9
	banded := make([]float32, batch*outputs)
	for i := range banded {
		banded[i] = guard
	}
	MatMatBF16Rows(w, x, outputs, inputs, batch, banded, from, upto)

	for l := 0; l < batch; l++ {
		for o := 0; o < outputs; o++ {
			at := l*outputs + o
			if o < from || o >= upto {
				if banded[at] != guard {
					t.Fatalf("row %d of vector %d is outside [%d,%d) and was written: %v",
						o, l, from, upto, banded[at])
				}
				continue
			}
			d := math.Abs(float64(banded[at] - full[at]))
			if d > 1e-4*(1+math.Abs(float64(full[at]))) {
				t.Fatalf("row %d of vector %d: %v against %v", o, l, banded[at], full[at])
			}
		}
	}
}
