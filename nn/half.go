package nn

import "math"

// floatToHalf rounds a float32 to the nearest IEEE binary16, ties to even.
// The Q8_0 scale is stored as an fp16 in ggml, so an activation quantized here
// must lose the same precision, or the products disagree in the fourth digit.
func floatToHalf(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits >> 16 & 0x8000)
	exponent := int32(bits>>23&0xFF) - 127 + 15
	mantissa := bits & 0x7FFFFF

	switch {
	case bits&0x7FFFFFFF == 0:
		return sign
	case exponent >= 0x1F: // overflow, and infinities and NaNs
		if bits&0x7F800000 == 0x7F800000 && mantissa != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case exponent <= 0: // subnormal, or underflow to zero
		if exponent < -10 {
			return sign
		}
		mantissa |= 0x800000
		shift := uint32(14 - exponent)
		half := uint16(mantissa >> shift)
		remainder := mantissa & (1<<shift - 1)
		if remainder > 1<<(shift-1) || (remainder == 1<<(shift-1) && half&1 == 1) {
			half++
		}
		return sign | half
	}
	half := uint16(exponent)<<10 | uint16(mantissa>>13)
	if remainder := mantissa & 0x1FFF; remainder > 0x1000 || (remainder == 0x1000 && half&1 == 1) {
		half++
	}
	return sign | half
}

// RoundHalf returns v rounded to the nearest fp16 and widened back.
//
// It exists for the key-value cache: llama.cpp keeps its cache in fp16 by
// default, so a Go engine that keeps float32 there is not more accurate than
// the reference, it is different from it — by about one part in a thousand,
// which is the size a real mistake would have.
func RoundHalf(v float32) float32 { return halfToFloat(floatToHalf(v)) }

// RoundBF16 returns v rounded to the nearest bfloat16 and widened back, ties
// to even — the conversion ggml performs on the *activation* before a product
// against bfloat16 weights. Its kernel takes both operands in bf16, so an
// activation that keeps its float32 mantissa is not more accurate than the
// reference, it is a third of a percent away from it.
func RoundBF16(v float32) float32 {
	bits := math.Float32bits(v)
	if bits&0x7FFFFFFF > 0x7F800000 { // NaN, kept quiet
		return math.Float32frombits((bits>>16 | 64) << 16)
	}
	return math.Float32frombits((bits + 0x7FFF + (bits >> 16 & 1)) &^ 0xFFFF)
}
