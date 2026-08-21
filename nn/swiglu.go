package nn

// The gated half of a feed forward, as ggml computes it.
//
// ggml_vec_swiglu_f32 applies SiLU to the gate and multiplies by the up
// projection in one pass, and its SiLU goes through ggml_v_expf — the ARM
// optimized exponential, accurate to about one and a half units in the last
// place, not the exact one. Both of those are reproduced here, and for the two
// different reasons that usually pull the other way.
//
// The accuracy is a wash: measured against the recorded ffn_out of a Qwen
// block, math.Exp in float64 and this polynomial differ by about a part in ten
// million, two orders below the float32 summation order already between this
// engine and llama.cpp. So matching ggml costs nothing there.
//
// The speed is not a wash. math.Exp is a float64 routine with a table lookup
// and two conversions around it; this is nine multiply-adds on float32. In a
// prompt of sixty-four positions the activation was 6.9% of the whole reading.
//
// nn.SiLU keeps math.Exp and is untouched: pockettts uses it, against a PyTorch
// reference rather than a ggml one, and its accuracy question is its own.

import "math"

// expf is ggml_v_expf, one value at a time.
//
// Adapted from the same Arm Limited routine llama.cpp adapted, which is why the
// constants are written as hexadecimal floats: they are bit patterns chosen for
// the polynomial, and a decimal rendering of them would be a transcription
// rather than the thing itself.
func expf(x float32) float32 {
	const (
		// 1.5 * 2^23. Adding it forces the round-to-nearest that puts the
		// integer part of x*log2(e) in the low mantissa bits.
		shift = float32(0x1.8p23)
		log2e = float32(0x1.715476p+0)
		// ln 2, split so that n*ln2 is exact in the high part.
		ln2hi = float32(0x1.62e4p-1)
		ln2lo = float32(0x1.7f7d1cp-20)

		c0 = float32(0x1.ffffecp-1)
		c1 = float32(0x1.fffdb6p-2)
		c2 = float32(0x1.555e66p-3)
		c3 = float32(0x1.573e2ep-5)
		c4 = float32(0x1.0e4020p-7)
	)

	z := x*log2e + shift
	n := z - shift

	// Outside this range the routine needs the two-step scaling llama.cpp
	// keeps for it. Those inputs are the ones where the result has already
	// stopped mattering — exp(-88) is a denormal, exp(88) an infinity — so the
	// exact routine answers them instead, which costs nothing at a frequency
	// of never.
	if n > 126 || n < -126 {
		return float32(math.Exp(float64(x)))
	}

	// ggml evaluates this with fused multiply-adds. Reproducing that through
	// math.FMA and a pair of conversions was tried and dropped: it left every
	// value bit for bit where it already was — the polynomial is well enough
	// conditioned that the second rounding never shows — and cost three and a
	// third times the time.
	b := x - n*ln2hi - n*ln2lo
	// 2^n, built by moving n into the exponent field.
	k := math.Float32frombits(math.Float32bits(z)<<23 + math.Float32bits(1))

	u := b * b
	j := ((c4*b+c3)*u+(c2*b+c1))*u + c0*b
	return j*k + k
}

// SwiGLURange writes silu(gate) * up into gate, over one stretch, on the
// caller's thread: the section that computed those values applies it before its
// barrier.
func SwiGLURange(gate, up []float32, start, end int) {
	for i := start; i < end; i++ {
		g := gate[i]
		gate[i] = g / (1 + expf(-g)) * up[i]
	}
}
