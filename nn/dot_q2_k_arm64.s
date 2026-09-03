//go:build arm64

#include "textflag.h"

// The integer half of the Q2_K product, on NEON.
//
// A superblock is 84 bytes and holds 256 weights: sixteen bytes each packing a
// four-bit scale and a four-bit minimum, sixty-four bytes of two-bit quants,
// and the two fp16 those nibbles are measured in. Only the scales reach this
// kernel — the minima are a group's, the activation carries a sum for every
// group of sixteen, and their whole product is sixteen multiplies the caller
// does in Go without ever touching a weight.
//
// The quants are walked as Q3_K's are: four chunks of thirty-two share a window
// of thirty-two bytes at four bit positions, and the two halves of a chunk are
// sixteen bytes apart inside it. A half is sixteen weights, which is a register
// exactly and the span of one scale, so the four SDOT lanes fold to the group's
// sum.
//
// Everything accumulates in int32. The largest a superblock can reach is
// 3 x 127 x 256 against a scale of 15, which is a five-hundredth of what an
// int32 holds, so the sum is exact and this form and the portable one agree bit
// for bit.

// MAG isolates one chunk's two bits at SH; MAG0 is the chunk that sits at bit
// nought, where NEON has no shift to offer — USHR takes one to eight.
#define MAG(QREG, SH, DST) \
	VUSHR $SH, QREG, DST   \
	VAND  V1.B16, DST, DST

#define MAG0(QREG, DST) \
	VAND V1.B16, QREG, DST

// GROUPS folds a chunk's two groups against their activations and their own
// scales, and adds both to the superblock's running sum in R10.
#define GROUPS(AOFF, SOFF)                                 \
	ADD  $(AOFF), R1, R5                               \
	VLD1 (R5), [V9.B16]                                \
	ADD  $(AOFF+16), R1, R5                            \
	VLD1 (R5), [V10.B16]                               \
	                                                   \
	VEOR V13.B16, V13.B16, V13.B16                     \
	VEOR V14.B16, V14.B16, V14.B16                     \
	                                                   \
	WORD $0x4e8994ad  /* SDOT V13.4S, V5.16B,  V9.16B  */ \
	WORD $0x4e8a950e  /* SDOT V14.4S, V8.16B, V10.16B  */ \
	                                                   \
	VADDV V13.S4, V21                                  \
	FMOVS F21, R5                                      \
	MOVB  (SOFF)(R4), R6                               \
	MULW  R6, R5, R5                                   \
	ADDW  R5, R10                                      \
	                                                   \
	VADDV V14.S4, V21                                  \
	FMOVS F21, R5                                      \
	MOVB  (SOFF+1)(R4), R6                             \
	MULW  R6, R5, R5                                   \
	ADDW  R5, R10

// func q2_kSumsNEON(w *byte, q *int8, scales *int8, blocks int, sums *int32)
//
// w advances 84 bytes a superblock, q advances 256 and scales 16. sums receives
// one int32 a superblock: the scaled sum of the magnitudes, before the minima
// and before any float.
TEXT ·q2_kSumsNEON(SB), NOSPLIT, $0-40
	MOVD w+0(FP), R0
	MOVD q+8(FP), R1
	MOVD scales+16(FP), R4
	MOVD blocks+24(FP), R2
	MOVD sums+32(FP), R3

	CBZ   R2, done
	VMOVI $3, V1.B16

loop:
	MOVW $0, R10

	ADD  $16, R0, R5
	VLD1 (R5), [V2.B16]     // chunks nought to three, low half
	ADD  $32, R0, R5
	VLD1 (R5), [V3.B16]     // and their high half
	ADD  $48, R0, R5
	VLD1 (R5), [V6.B16]     // chunks four to seven, low half
	ADD  $64, R0, R5
	VLD1 (R5), [V7.B16]     // and their high half

	MAG0(V2.B16, V5.B16)
	MAG0(V3.B16, V8.B16)
	GROUPS(0, 0)

	MAG(V2.B16, 2, V5.B16)
	MAG(V3.B16, 2, V8.B16)
	GROUPS(32, 2)

	MAG(V2.B16, 4, V5.B16)
	MAG(V3.B16, 4, V8.B16)
	GROUPS(64, 4)

	MAG(V2.B16, 6, V5.B16)
	MAG(V3.B16, 6, V8.B16)
	GROUPS(96, 6)

	MAG0(V6.B16, V5.B16)
	MAG0(V7.B16, V8.B16)
	GROUPS(128, 8)

	MAG(V6.B16, 2, V5.B16)
	MAG(V7.B16, 2, V8.B16)
	GROUPS(160, 10)

	MAG(V6.B16, 4, V5.B16)
	MAG(V7.B16, 4, V8.B16)
	GROUPS(192, 12)

	MAG(V6.B16, 6, V5.B16)
	MAG(V7.B16, 6, V8.B16)
	GROUPS(224, 14)

	MOVW R10, (R3)

	ADD  $84, R0
	ADD  $256, R1
	ADD  $16, R4
	ADD  $4, R3
	SUB  $1, R2
	CBNZ R2, loop

done:
	RET
