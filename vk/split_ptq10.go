package vk

// PTQ1_0 on the card.
//
// A Prism ternary format packed at group size 128: 24 bytes qs (120 trits),
// 2 bytes qh (8 trits), and one fp16 scale (bytes 26..27).
//
// A block is 28 bytes, which is seven whole 32-bit words:
//   words 0..5: qs[0..23]
//   word 6:     qh[0..1] at bytes 0..1, d fp16 at bytes 2..3
//
// Because every block is already an exact whole number of 32-bit words, the
// file layout is already word-aligned on the card. splitPTQ1_0 performs a copy
// with length validation.

import "github.com/ThiraSoft/golem/nn"

// rowBytesPTQ1_0 is what one row of that many inputs occupies on the card,
// which is what it occupies in the file: 28 bytes per 128 weights.
func rowBytesPTQ1_0(cols int) int {
	return cols / nn.TernaryBlock * 28
}

// splitPTQ1_0 verifies and copies PTQ1_0 data to the card layout.
func splitPTQ1_0(src []byte, rows, cols int) []byte {
	stride := rowBytesPTQ1_0(cols)
	dst := make([]byte, rows*stride)
	copy(dst, src[:rows*stride])
	return dst
}

// ptq10ByteAndPower returns the byte index (within the 28-byte block) and
// the pow3 power index (0..4) for weight index v (0..127) within a 128-block.
// Byte indices 0..23 map to qs[0..23], and indices 24..25 map to qh[0..1].
func ptq10ByteAndPower(v int) (byteIndex int, powerIndex int) {
	if v < 80 {
		return v % 16, v / 16
	} else if v < 120 {
		return 16 + (v-80)%8, (v - 80) / 8
	}
	return 24 + (v-120)%2, (v - 120) / 2
}
