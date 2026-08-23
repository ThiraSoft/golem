//go:build arm64

#include "textflag.h"

// The integer half of the Q4_0 product, on NEON.
//
// One block is 18 bytes: an fp16 scale, then 32 weights packed two to a byte,
// the low nibble holding weight j and the high nibble weight j+16. Against a
// block of the Q8_0 activation that is thirty-two products, and their sum is
// what this writes out — one int32 per block, nothing scaled and nothing
// converted.
//
// Splitting it there is what makes the port safe. The sum of a block cannot
// overflow — fifteen times a hundred and twenty-seven, thirty-two times over,
// is sixty thousand against two billion — so integer addition is exact and
// associative, and it does not matter that SDOT accumulates in four lanes where
// the portable form accumulates in one. The two agree bit for bit, and the test
// beside this file demands exactly that rather than a tolerance.
//
// What is left in Go is the fp16 scale, the two multiplies and the running sum.
// That half needs SCVTF and FCVTHS on vectors, which would be two more
// hand-encoded words for arithmetic that costs four operations a block against
// the thirty-two removed here. It is not worth the encoding risk.
//
// The only word is SDOT; nn/encodings_arm64.s explains it and
// nn/encodings_arm64_test.go proves it.

// func q4_0SumsNEON(w *byte, q *int8, blocks, qStride int, sums *int32)
//
// w advances 18 bytes a block. q advances qStride bytes a block, which is the
// batch stride in blocks times thirty-two. sums receives one int32 a block.
TEXT ·q4_0SumsNEON(SB), NOSPLIT, $0-40
	MOVD w+0(FP), R0
	MOVD q+8(FP), R1
	MOVD blocks+16(FP), R2
	MOVD qStride+24(FP), R3
	MOVD sums+32(FP), R4

	CBZ  R2, done
	VMOVI $15, V6.B16              // the low-nibble mask

loop:
	// The nibbles sit two bytes past the block, after the scale.
	ADD  $2, R0, R5
	VLD1 (R5), [V1.B16]            // sixteen bytes: weights 0..31, packed
	VLD1 (R1), [V4.B16, V5.B16]    // q[0..15] and q[16..31]

	VAND  V6.B16, V1.B16, V2.B16   // low nibbles:  weights 0..15
	VUSHR $4, V1.B16, V3.B16       // high nibbles: weights 16..31

	// Zeroed every block: SDOT accumulates, and each block's sum stands alone.
	VEOR V0.B16, V0.B16, V0.B16

	WORD $0x4e849440               // SDOT V0.4S, V2.16B, V4.16B
	WORD $0x4e859460               // SDOT V0.4S, V3.16B, V5.16B

	VADDV V0.S4, V7                // the four lanes into one
	FMOVS F7, R6
	MOVW  R6, (R4)

	ADD $18, R0
	ADD R3, R1
	ADD $4, R4
	SUB $1, R2
	CBNZ R2, loop

done:
	RET
