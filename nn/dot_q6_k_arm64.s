//go:build arm64

#include "textflag.h"

// The integer half of the Q6_K product, on NEON.
//
// A superblock is 210 bytes and holds 256 weights: 128 bytes of low nibbles, 64
// bytes carrying the two high bits of each weight two bits at a time, sixteen
// signed group scales, and one fp16 scale over the lot. A weight is six bits
// unsigned, in 0..63, and is never recentred here — the portable form takes the
// 32 out afterwards as 32 x scale x sum(q) per group, which is why the
// activation carries its group sums and why this kernel does not see them.
//
// The awkward part is the order. Reading i from 0 to 31, the four weights a
// byte pair yields are l[i]&15 | (h[i]&3)<<4, l[i+32]&15 | (h[i]>>2&3)<<4,
// l[i]>>4 | (h[i]>>4&3)<<4 and l[i+32]>>4 | (h[i]>>6&3)<<4 — four planes,
// against activations 0, 32, 64 and 96 further on, each with its own group
// scale. So the kernel walks sixteen at a time: one plane of sixteen weights
// fills a register exactly, and its four SDOT lanes fold to the group's sum.
//
// Everything accumulates in int32, as the portable form does. The largest a
// superblock can reach is 63 x 127 x 256 against a scale of 127, which is a
// quarter of what an int32 holds, so the sum is exact and the two forms agree
// bit for bit. The recentring, the fp16 scale and the running total stay in Go.

// One pass over sixteen weights of each of the four planes.
//
// LOFF is the byte offset into the low nibbles, HOFF into the high bits, AOFF
// into the activation and SOFF into the group scales. The four scales a pass
// needs are SOFF, SOFF+2, SOFF+4 and SOFF+6, which is how ggml lays them out.
//
// R0 is the superblock, R1 the activation, R10 the running sum. R5 and R6 are
// scratch. V0 holds 0x0f and V1 holds 0x03 throughout.
#define PASS(LOFF, HOFF, AOFF, SOFF)                       \
	ADD  $(LOFF), R0, R5                               \
	VLD1 (R5), [V2.B16]                                \
	ADD  $(LOFF+32), R0, R5                            \
	VLD1 (R5), [V3.B16]                                \
	ADD  $(128+HOFF), R0, R5                           \
	VLD1 (R5), [V4.B16]                                \
	                                                   \
	VAND  V0.B16, V2.B16, V5.B16                       \
	VAND  V1.B16, V4.B16, V17.B16                      \
	VSHL  $4, V17.B16, V17.B16                         \
	VORR  V17.B16, V5.B16, V5.B16                      \
	                                                   \
	VAND  V0.B16, V3.B16, V6.B16                       \
	VUSHR $2, V4.B16, V18.B16                          \
	VAND  V1.B16, V18.B16, V18.B16                     \
	VSHL  $4, V18.B16, V18.B16                         \
	VORR  V18.B16, V6.B16, V6.B16                      \
	                                                   \
	VUSHR $4, V2.B16, V7.B16                           \
	VUSHR $4, V4.B16, V19.B16                          \
	VAND  V1.B16, V19.B16, V19.B16                     \
	VSHL  $4, V19.B16, V19.B16                         \
	VORR  V19.B16, V7.B16, V7.B16                      \
	                                                   \
	VUSHR $4, V3.B16, V8.B16                           \
	VUSHR $6, V4.B16, V20.B16                          \
	VSHL  $4, V20.B16, V20.B16                         \
	VORR  V20.B16, V8.B16, V8.B16                      \
	                                                   \
	ADD  $(AOFF), R1, R5                               \
	VLD1 (R5), [V9.B16]                                \
	ADD  $(AOFF+32), R1, R5                            \
	VLD1 (R5), [V10.B16]                               \
	ADD  $(AOFF+64), R1, R5                            \
	VLD1 (R5), [V11.B16]                               \
	ADD  $(AOFF+96), R1, R5                            \
	VLD1 (R5), [V12.B16]                               \
	                                                   \
	VEOR V13.B16, V13.B16, V13.B16                     \
	VEOR V14.B16, V14.B16, V14.B16                     \
	VEOR V15.B16, V15.B16, V15.B16                     \
	VEOR V16.B16, V16.B16, V16.B16                     \
	                                                   \
	WORD $0x4e8994ad  /* SDOT V13.4S, V5.16B,  V9.16B  */ \
	WORD $0x4e8a94ce  /* SDOT V14.4S, V6.16B, V10.16B  */ \
	WORD $0x4e8b94ef  /* SDOT V15.4S, V7.16B, V11.16B  */ \
	WORD $0x4e8c9510  /* SDOT V16.4S, V8.16B, V12.16B  */ \
	                                                   \
	VADDV V13.S4, V21                                  \
	FMOVS F21, R5                                      \
	MOVB  (192+SOFF)(R0), R6                           \
	MULW  R6, R5, R5                                   \
	ADDW  R5, R10                                      \
	                                                   \
	VADDV V14.S4, V21                                  \
	FMOVS F21, R5                                      \
	MOVB  (192+SOFF+2)(R0), R6                         \
	MULW  R6, R5, R5                                   \
	ADDW  R5, R10                                      \
	                                                   \
	VADDV V15.S4, V21                                  \
	FMOVS F21, R5                                      \
	MOVB  (192+SOFF+4)(R0), R6                         \
	MULW  R6, R5, R5                                   \
	ADDW  R5, R10                                      \
	                                                   \
	VADDV V16.S4, V21                                  \
	FMOVS F21, R5                                      \
	MOVB  (192+SOFF+6)(R0), R6                         \
	MULW  R6, R5, R5                                   \
	ADDW  R5, R10

// func q6_kSumsNEON(w *byte, q *int8, blocks int, sums *int32)
//
// w advances 210 bytes a superblock, q advances 256. sums receives one int32 a
// superblock: the whole product before recentring and before any scale that is
// not a group's own.
TEXT ·q6_kSumsNEON(SB), NOSPLIT, $0-32
	MOVD w+0(FP), R0
	MOVD q+8(FP), R1
	MOVD blocks+16(FP), R2
	MOVD sums+24(FP), R3

	CBZ   R2, done
	VMOVI $15, V0.B16
	VMOVI $3, V1.B16

loop:
	MOVW $0, R10

	// half 0, group 0 and 1, then half 1. The offsets are the portable form's
	// l[half*64], h[half*32], a[half*128] and s[half*8], with the group's
	// sixteen added on.
	PASS(0, 0, 0, 0)
	PASS(16, 16, 16, 1)
	PASS(64, 32, 128, 8)
	PASS(80, 48, 144, 9)

	MOVW R10, (R3)

	ADD $210, R0
	ADD $256, R1
	ADD $4, R3
	SUB $1, R2
	CBNZ R2, loop

done:
	RET
