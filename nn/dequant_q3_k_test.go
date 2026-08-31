package nn

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestDequantizeQ3_KMatchesReference builds one superblock whose every field is
// known and checks the weights ggml's traversal order produces. The layout is
// hmask[32], qs[64], scales[12], d — and the quant is a two-bit value from qs
// with a third bit from hmask, recentred by 4 and scaled by a six-bit signed
// scale recentred by 32.
func TestDequantizeQ3_KMatchesReference(t *testing.T) {
	block := make([]byte, q3_kBlockBytes)
	// hmask all ones: every weight gets its high bit, so q = low | 4.
	for i := 0; i < 32; i++ {
		block[i] = 0xFF
	}
	// qs: weight 0 takes the low two bits of qs[0], weight 1 of qs[1], ...
	// Set qs[0] = 0b01 so weight 0's low bits are 1 and its quant is 1|4 = 5,
	// which recentred by 4 is +1.
	block[32] = 0x01
	// scales: twelve bytes holding sixteen six-bit values. All zero means every
	// scale is 0-32 = -32.
	// d = 1.0
	binary.LittleEndian.PutUint16(block[108:], floatToHalf(1))

	out := make([]float32, SuperBlock)
	DequantizeQ3_K(block, SuperBlock, out)

	// weight 0: d · scale · (q - 4) = 1 · (-32) · (5 - 4) = -32
	if got := out[0]; math.Abs(float64(got)-(-32)) > 1e-4 {
		t.Fatalf("weight 0 = %v, want -32", got)
	}
	// weight 1: qs[1] = 0, so q = 0|4 = 4, recentred to 0, so the weight is 0.
	if got := out[1]; got != 0 {
		t.Fatalf("weight 1 = %v, want 0", got)
	}
}

// TestQ3_KRowGeometry is what the loader will divide by.
func TestQ3_KRowGeometry(t *testing.T) {
	if q3_kBlockBytes != 110 {
		t.Fatalf("a Q3_K superblock is 110 bytes, not %d", q3_kBlockBytes)
	}
}
