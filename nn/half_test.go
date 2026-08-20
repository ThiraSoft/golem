package nn

import "testing"

// TestHalfRoundTrip checks that every representable binary16 survives the trip
// through float32 and back. That covers subnormals, the two zeros, and the
// boundary where the exponent field saturates — the three places a hand-written
// conversion goes wrong.
func TestHalfRoundTrip(t *testing.T) {
	for bits := 0; bits < 1<<16; bits++ {
		h := uint16(bits)
		exponent, mantissa := h>>10&0x1F, h&0x3FF
		if exponent == 0x1F && mantissa != 0 {
			continue // NaN payloads are not preserved, and need not be
		}
		if got := floatToHalf(halfToFloat(h)); got != h {
			t.Fatalf("%#04x came back as %#04x (%g)", h, got, halfToFloat(h))
		}
	}
}

// TestFloatToHalfRounds checks the ties, which is where the Q8_0 scale lands
// often enough to matter.
func TestFloatToHalfRounds(t *testing.T) {
	cases := []struct {
		f    float32
		want uint16
	}{
		{0, 0x0000},
		{1, 0x3C00},
		{-2, 0xC000},
		{65504, 0x7BFF},         // the largest finite binary16
		{131008, 0x7C00},        // beyond it: infinity
		{6.0e-8, 0x0001},        // the smallest subnormal
		{1.0e-9, 0x0000},        // below it: zero
		{1.00048828125, 0x3C00}, // halfway between 0x3C00 and 0x3C01: to the even one
		{1.00146484375, 0x3C02}, // halfway the other way, again to the even one
	}
	for _, c := range cases {
		if got := floatToHalf(c.f); got != c.want {
			t.Errorf("floatToHalf(%g) = %#04x, want %#04x", c.f, got, c.want)
		}
	}
}
