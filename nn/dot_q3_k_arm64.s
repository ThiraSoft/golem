//go:build arm64

#include "textflag.h"

// The integer half of the Q3_K product, on NEON.
//
// A superblock is 110 bytes and holds 256 weights: 32 bytes carrying one bit of
// each weight, 64 bytes carrying the other two, twelve bytes of six-bit group
// scales and one fp16 scale over the lot. A weight is three bits unsigned, in
// 0..7, and is never recentred here — the caller takes the four out afterwards
// as 4 x scale x sum(q) per group, which is why the activation carries its
// group sums and why this kernel does not see them.
//
// The scales arrive already unpacked and already signed, sixteen bytes to a
// superblock, because the six-bit packing is four masked shifts of three
// different bytes and it is the one part of this format that is cheaper in Go
// than in a register. q3_kScales is that unpacking, and it has a fixture behind
// it; nothing here has to trust a second copy of it.
//
// The order is the awkward part. The first thirty-two bytes of qs carry four
// chunks of thirty-two weights at four bit positions, and the second thirty-two
// carry the other four; the third bits of all eight chunks are eight bit
// positions of the same thirty-two bytes of hmask. So a group of sixteen
// weights is sixteen contiguous bytes of one qs window and sixteen of hmask,
// both read at the chunk's own two shifts — which is a register exactly, and
// the four SDOT lanes fold to the group's sum.
//
// Everything accumulates in int32. The largest a superblock can reach is
// 7 x 127 x 256 against a scale of 32, which is a fortieth of what an int32
// holds, so the sum is exact and this form and the portable one agree bit for
// bit.

// MAG builds one group's sixteen magnitudes: the chunk's two low bits out of a
// qs window at SH, and its third bit out of hmask at HB, into bit two.
//
// Both are byte-lane shifts done as halfword shifts, which drags the
// neighbouring byte's bits into the top of each. The masks are what makes that
// harmless: after a shift of SH the wanted pair is at bits 0 and 1 and the
// intruders are at 6 and 7, and V1 keeps only the pair. The third bit is the
// same argument at one bit and V0.
#define MAG(QREG, HREG, SH, HB, DST) \
	VUSHR $SH, QREG, DST      \
	VAND  V1.B16, DST, DST    \
	VUSHR $HB, HREG, V20.B16  \
	VAND  V0.B16, V20.B16, V20.B16 \
	VSHL  $2, V20.B16, V20.B16 \
	VORR  V20.B16, DST, DST

// MAGS0 is MAG for the chunk that sits at bit nought of its window, and MAG00
// for the one that sits at bit nought of hmask as well. A shift of zero is not
// an encoding NEON has — USHR takes one to eight — so the two cases where the
// shift would be nought drop the instruction instead.
#define MAGS0(QREG, HREG, HB, DST) \
	VAND  V1.B16, QREG, DST   \
	VUSHR $HB, HREG, V20.B16  \
	VAND  V0.B16, V20.B16, V20.B16 \
	VSHL  $2, V20.B16, V20.B16 \
	VORR  V20.B16, DST, DST

#define MAG00(QREG, HREG, DST) \
	VAND V1.B16, QREG, DST   \
	VAND V0.B16, HREG, V20.B16 \
	VSHL $2, V20.B16, V20.B16 \
	VORR V20.B16, DST, DST

// GROUPS folds two groups — the low sixteen weights of a chunk and the high
// sixteen — against their activations and their own scales, and adds both to
// the superblock's running sum in R10.
//
// V5 and V8 are the magnitudes the MAG macros left. AOFF is where the chunk's
// thirty-two activations begin and SOFF where its two scales do.
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

// func q3_kSumsNEON(w *byte, q *int8, scales *int8, blocks int, sums *int32)
//
// w advances 110 bytes a superblock, q advances 256 and scales 16. sums
// receives one int32 a superblock: the whole product before recentring and
// before any scale that is not a group's own.
TEXT ·q3_kSumsNEON(SB), NOSPLIT, $0-40
	MOVD w+0(FP), R0
	MOVD q+8(FP), R1
	MOVD scales+16(FP), R4
	MOVD blocks+24(FP), R2
	MOVD sums+32(FP), R3

	CBZ   R2, done
	VMOVI $1, V0.B16
	VMOVI $3, V1.B16

loop:
	MOVW $0, R10

	ADD  $0, R0, R5
	VLD1 (R5), [V4.B16]     // hmask, the first sixteen weights of every chunk
	ADD  $16, R0, R5
	VLD1 (R5), [V22.B16]    // and the second sixteen
	ADD  $32, R0, R5
	VLD1 (R5), [V2.B16]     // the low pairs, chunks nought to three, low half
	ADD  $48, R0, R5
	VLD1 (R5), [V3.B16]     // and their high half
	ADD  $64, R0, R5
	VLD1 (R5), [V6.B16]     // chunks four to seven, low half
	ADD  $80, R0, R5
	VLD1 (R5), [V7.B16]     // and their high half

	MAG00(V2.B16, V4.B16, V5.B16)
	MAG00(V3.B16, V22.B16, V8.B16)
	GROUPS(0, 0)

	MAG(V2.B16, V4.B16, 2, 1, V5.B16)
	MAG(V3.B16, V22.B16, 2, 1, V8.B16)
	GROUPS(32, 2)

	MAG(V2.B16, V4.B16, 4, 2, V5.B16)
	MAG(V3.B16, V22.B16, 4, 2, V8.B16)
	GROUPS(64, 4)

	MAG(V2.B16, V4.B16, 6, 3, V5.B16)
	MAG(V3.B16, V22.B16, 6, 3, V8.B16)
	GROUPS(96, 6)

	MAGS0(V6.B16, V4.B16, 4, V5.B16)
	MAGS0(V7.B16, V22.B16, 4, V8.B16)
	GROUPS(128, 8)

	MAG(V6.B16, V4.B16, 2, 5, V5.B16)
	MAG(V7.B16, V22.B16, 2, 5, V8.B16)
	GROUPS(160, 10)

	MAG(V6.B16, V4.B16, 4, 6, V5.B16)
	MAG(V7.B16, V22.B16, 4, 6, V8.B16)
	GROUPS(192, 12)

	MAG(V6.B16, V4.B16, 6, 7, V5.B16)
	MAG(V7.B16, V22.B16, 6, 7, V8.B16)
	GROUPS(224, 14)

	MOVW R10, (R3)

	ADD  $110, R0
	ADD  $256, R1
	ADD  $16, R4
	ADD  $4, R3
	SUB  $1, R2
	CBNZ R2, loop

done:
	RET
