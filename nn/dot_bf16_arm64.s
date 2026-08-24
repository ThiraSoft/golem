//go:build arm64

#include "textflag.h"

// The bfloat16 product on NEON.
//
// This is the kernel the speech engine runs on. Pocket TTS keeps its weights in
// bfloat16 and synthesizes one frame at a time, so every projection it makes is
// a matrix-vector product taken a row at a time through here — the batch
// kernels never see it, and neither does the quantized path the transformer
// uses. Until this file existed, arm64 ran the whole of that in portable Go.
//
// Widening costs nothing here. A bfloat16 is a float32 with its low mantissa
// cut off, so putting one back is placing those sixteen bits in the high half
// of a word and zeroing the low half — which is exactly what an interleave with
// zero does. ZIP1 takes the low four halves of the source, ZIP2 the high four,
// and each writes a word whose top is a weight and whose bottom is nothing. Two
// instructions widen eight weights, and both are ones Go's assembler spells, so
// there is no hand-encoded word in this file.
//
// The four accumulators and the lane each element lands in are copied from
// float_kernels_arm64.s. That is not for bit-parity — nothing here demands it,
// and the tolerance is a relative 1e-4 — but the folding code in Go is then the
// same three lines, and one shape verified once is worth more than two.

// func dotBF16LanesNEON(row *uint16, x *float32, n int, out *float32)
//
// out receives the four accumulators of four lanes, then the scalar remainder:
// seventeen floats, the same block its float32 and fp16 counterparts write.
TEXT ·dotBF16LanesNEON(SB), NOSPLIT, $0-32
	MOVD row+0(FP), R0
	MOVD x+8(FP), R1
	MOVD n+16(FP), R2
	MOVD out+24(FP), R3

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	// The zero half of every interleave.
	VEOR V16.B16, V16.B16, V16.B16

	MOVD R2, R4
	LSR  $4, R4
	CBZ  R4, btail4

bwide:
	VLD1.P 16(R0), [V12.H8]
	VZIP1  V12.H8, V16.H8, V8.H8
	VZIP2  V12.H8, V16.H8, V9.H8
	VLD1.P 16(R1), [V4.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	VLD1.P 16(R1), [V5.S4]
	VFMLA  V9.S4, V5.S4, V1.S4

	VLD1.P 16(R0), [V13.H8]
	VZIP1  V13.H8, V16.H8, V10.H8
	VZIP2  V13.H8, V16.H8, V11.H8
	VLD1.P 16(R1), [V6.S4]
	VFMLA  V10.S4, V6.S4, V2.S4
	VLD1.P 16(R1), [V7.S4]
	VFMLA  V11.S4, V7.S4, V3.S4

	SUB  $1, R4
	CBNZ R4, bwide

btail4:
	AND $15, R2, R5
	LSR $2, R5, R6
	CBZ R6, btail1

bfour:
	VLD1.P 8(R0), [V12.D1]
	VZIP1  V12.H8, V16.H8, V8.H8
	VLD1.P 16(R1), [V4.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	SUB    $1, R6
	CBNZ   R6, bfour

btail1:
	FMOVS $(0.0), F14
	AND   $3, R2, R7
	CBZ   R7, bstore

bone:
	MOVHU.P 2(R0), R8
	LSL     $16, R8
	FMOVS   R8, F13
	FMOVS.P 4(R1), F12
	FMADDS  F13, F14, F12, F14
	SUB     $1, R7
	CBNZ    R7, bone

bstore:
	VST1  [V0.S4, V1.S4, V2.S4, V3.S4], (R3)
	FMOVS F14, 64(R3)
	RET
