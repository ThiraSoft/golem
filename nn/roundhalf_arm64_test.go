//go:build arm64

package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The NEON rounding against RoundHalf, exactly.
//
// FCVTN rounds to nearest even and so does floatToHalf, so there is no latitude
// to allow here: any difference is a difference in behaviour, and this writes
// the key-value cache. The values are the ones random floats never reach —
// subnormals at both ends, the point where fp16 overflows to infinity, the
// infinities themselves and a NaN.
func TestRoundHalfNEONMatchesRoundHalf(t *testing.T) {
	awkward := []float32{
		0, float32(math.Copysign(0, -1)),
		5.96e-8, -5.96e-8, // the smallest fp16 subnormal
		6.0e-8, 3.0e-8, // below it, rounding to zero or to it
		6.104e-5, -6.104e-5, // the smallest normal
		65504, -65504, // the largest finite fp16
		65519, 65520, // just under and over the overflow point
		65535, 1e30, -1e30,
		float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()),
		1, -1, 0.5, 1.0009765625, 2049, 0.333333,
	}

	rng := rand.New(rand.NewSource(71))
	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 64, 129} {
		v := make([]float32, n)
		want := make([]float32, n)
		for i := range v {
			if i < len(awkward) {
				v[i] = awkward[i%len(awkward)]
			} else {
				v[i] = float32(rng.NormFloat64() * 100)
			}
			want[i] = RoundHalf(v[i])
		}
		RoundHalfRange(v)

		for i := range v {
			got, wanted := v[i], want[i]
			if math.IsNaN(float64(got)) && math.IsNaN(float64(wanted)) {
				continue
			}
			if math.Float32bits(got) != math.Float32bits(wanted) {
				t.Fatalf("n=%d element %d: NEON %v (%#x), RoundHalf %v (%#x)",
					n, i, got, math.Float32bits(got), wanted, math.Float32bits(wanted))
			}
		}
	}
}

// Every value the whole fp16 range can express, and every float32 that rounds
// near a boundary between two of them: a sweep dense enough that a wrong
// rounding mode shows up rather than hides between the samples.
func TestRoundHalfNEONOverTheWholeRange(t *testing.T) {
	const batch = 64
	v := make([]float32, batch)
	want := make([]float32, batch)

	for base := 0; base < 1<<16; base += batch {
		for i := 0; i < batch; i++ {
			h := uint16(base + i)
			// The midpoint between this fp16 and the next, which is exactly
			// where round-to-nearest-even and round-half-away disagree.
			v[i] = (halfToFloat(h) + halfToFloat(h+1)) / 2
			want[i] = RoundHalf(v[i])
		}
		RoundHalfRange(v)
		for i := 0; i < batch; i++ {
			if math.IsNaN(float64(v[i])) && math.IsNaN(float64(want[i])) {
				continue
			}
			if math.Float32bits(v[i]) != math.Float32bits(want[i]) {
				t.Fatalf("fp16 %#x midpoint: NEON %v, RoundHalf %v", uint16(base+i), v[i], want[i])
			}
		}
	}
}
