//go:build arm64

#include "textflag.h"

// The packed Q4_0 product, on NEON.
//
// This is the kernel the four-row layout exists for, and it is the one place in
// the port where nothing has to be worked around. A chunk's low nibbles are
// already [row0: four inputs][row1][row2][row3]; the four activations they all
// meet are one 32-bit word, and VLD1R puts that word in every lane. So one SDOT
// accumulates four rows at once, each in its own lane, and the high nibbles do
// the same for the next four inputs. No shuffle, no horizontal add, no undoing
// of a layout meant for another machine.
//
// Sixteen bytes of weights and eight of activation a pass, four passes a block
// of thirty-two inputs, and the four lanes come out as the four rows' sums in
// order. Integers throughout, so it is exact: the float scales are applied in
// Go afterwards, as in the other two kernels here.
//
// The words are SDOT; nn/encodings_arm64.s explains the encoding and
// nn/encodings_arm64_test.go proves it.

// One chunk: sixteen bytes of weights against eight inputs, into V13.
//
// COFF is the chunk's offset into the group's nibbles, AOFF the matching offset
// into the activation. R0 is the group, R1 the activation, R5 scratch.
#define CHUNK(COFF, AOFF)                                  \
	ADD   $(8+COFF), R0, R5                            \
	VLD1  (R5), [V2.B16]                               \
	VAND  V0.B16, V2.B16, V5.B16                       \
	VUSHR $4, V2.B16, V6.B16                           \
	                                                   \
	ADD   $(AOFF), R1, R5                              \
	VLD1R (R5), [V9.S4]                                \
	ADD   $(AOFF+4), R1, R5                            \
	VLD1R (R5), [V10.S4]                               \
	                                                   \
	WORD $0x4e8994ad  /* SDOT V13.4S, V5.16B,  V9.16B */ \
	WORD $0x4e8a94cd  /* SDOT V13.4S, V6.16B, V10.16B */

// func packedQ4_0SumsNEON(w *byte, q *int8, blocks, qStride int, sums *int32)
//
// w advances 72 bytes a block, which is one group of four rows. q advances
// qStride bytes. sums receives four int32 a block, one per row, in row order.
TEXT ·packedQ4_0SumsNEON(SB), NOSPLIT, $0-40
	MOVD w+0(FP), R0
	MOVD q+8(FP), R1
	MOVD blocks+16(FP), R2
	MOVD qStride+24(FP), R3
	MOVD sums+32(FP), R4

	CBZ   R2, done
	VMOVI $15, V0.B16

loop:
	VEOR V13.B16, V13.B16, V13.B16

	CHUNK(0, 0)
	CHUNK(16, 8)
	CHUNK(32, 16)
	CHUNK(48, 24)

	VST1 [V13.B16], (R4)

	ADD $72, R0
	ADD R3, R1
	ADD $16, R4
	SUB $1, R2
	CBNZ R2, loop

done:
	RET
