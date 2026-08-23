//go:build arm64

#include "textflag.h"

// The dot product against a cache kept in fp16.
//
// This exists because of an invariant, not because of a speed. DotF32Half must
// give bit for bit what DotF32 gives on the same values widened — the test in
// dot_half_test.go says why: a mixture of experts takes a hard top-k over its
// router's logits, and a difference in the last bit picks a different expert.
// So accelerating the float32 product and leaving this one in Go is not an
// option; the two have to move together.
//
// Which is why the shape below is copied from float_kernels_arm64.s rather than
// chosen: the same four accumulators, the same sixteen elements a turn, the
// same element in the same lane of the same accumulator. Widening a half and
// then multiplying is exactly what the portable loop does, so with the
// arithmetic in the same order the two agree to the bit.
//
// FCVTL is the one hand-encoded word here. Go's assembler has FCVTHS for one
// scalar but nothing for four at a time.

// func dotF32HalfLanesNEON(a *float32, b *uint16, n int, out *float32)
//
// out receives the four accumulators of four lanes, then the scalar remainder —
// seventeen floats, as its float32 counterpart writes.
TEXT ·dotF32HalfLanesNEON(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	MOVD out+24(FP), R3

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	MOVD R2, R4
	LSR  $4, R4
	CBZ  R4, htail4

hwide:
	VLD1.P 8(R1), [V12.D1]
	WORD   $0x0e217988             // FCVTL V8.4S, V12.4H
	VLD1.P 16(R0), [V4.S4]
	VFMLA  V8.S4, V4.S4, V0.S4

	VLD1.P 8(R1), [V12.D1]
	WORD   $0x0e217988             // FCVTL V8.4S, V12.4H
	VLD1.P 16(R0), [V4.S4]
	VFMLA  V8.S4, V4.S4, V1.S4

	VLD1.P 8(R1), [V12.D1]
	WORD   $0x0e217988             // FCVTL V8.4S, V12.4H
	VLD1.P 16(R0), [V4.S4]
	VFMLA  V8.S4, V4.S4, V2.S4

	VLD1.P 8(R1), [V12.D1]
	WORD   $0x0e217988             // FCVTL V8.4S, V12.4H
	VLD1.P 16(R0), [V4.S4]
	VFMLA  V8.S4, V4.S4, V3.S4

	SUB  $1, R4
	CBNZ R4, hwide

htail4:
	AND $15, R2, R5
	LSR $2, R5, R6
	CBZ R6, htail1

hfour:
	VLD1.P 8(R1), [V12.D1]
	WORD   $0x0e217988             // FCVTL V8.4S, V12.4H
	VLD1.P 16(R0), [V4.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	SUB    $1, R6
	CBNZ   R6, hfour

htail1:
	FMOVS $(0.0), F14
	AND   $3, R2, R7
	CBZ   R7, hstore

hone:
	MOVHU.P 2(R1), R8
	FMOVS   R8, F13
	FCVTHS  F13, F13
	FMOVS.P 4(R0), F12
	FMADDS  F13, F14, F12, F14
	SUB     $1, R7
	CBNZ    R7, hone

hstore:
	VST1  [V0.S4, V1.S4, V2.S4, V3.S4], (R3)
	FMOVS F14, 64(R3)
	RET
