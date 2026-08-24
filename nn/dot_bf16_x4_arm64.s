//go:build arm64

#include "textflag.h"

// One row of bfloat16 weights against four activation columns.
//
// The batch kernel's reason for existing is that widening a weight costs the
// same whether one column or four are waiting for it. Here that is two ZIPs per
// eight weights, and they are paid once for thirty-two products instead of once
// for eight — the rest of the loop is loads and multiply-accumulates, which is
// what a kernel should be spending itself on.
//
// The widening is the interleave-with-zero described in dot_bf16_arm64.s. The
// four columns each get their own accumulator: they are independent chains, so
// there is no need for the four-accumulator trick the single-column kernel uses
// to break the dependency between successive additions.

// func dotBF16x4NEON(row *uint16, x *float32, stride, n int, out *float32)
//
// out receives four accumulators of four lanes, then the four scalar
// remainders: twenty floats, one column after another.
TEXT ·dotBF16x4NEON(SB), NOSPLIT, $0-40
	MOVD row+0(FP), R0
	MOVD x+8(FP), R1
	MOVD stride+16(FP), R9
	MOVD n+24(FP), R5
	MOVD out+32(FP), R6

	// The four columns, a stride of float32s apart.
	LSL $2, R9
	ADD R9, R1, R2
	ADD R9, R2, R3
	ADD R9, R3, R4

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V16.B16, V16.B16, V16.B16

	LSR $3, R5, R7
	CBZ R7, qtail4

qeight:
	VLD1.P 16(R0), [V12.H8]
	VZIP1  V12.H8, V16.H8, V8.H8
	VZIP2  V12.H8, V16.H8, V9.H8

	VLD1.P 16(R1), [V4.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	VLD1.P 16(R1), [V5.S4]
	VFMLA  V9.S4, V5.S4, V0.S4

	VLD1.P 16(R2), [V6.S4]
	VFMLA  V8.S4, V6.S4, V1.S4
	VLD1.P 16(R2), [V7.S4]
	VFMLA  V9.S4, V7.S4, V1.S4

	VLD1.P 16(R3), [V4.S4]
	VFMLA  V8.S4, V4.S4, V2.S4
	VLD1.P 16(R3), [V5.S4]
	VFMLA  V9.S4, V5.S4, V2.S4

	VLD1.P 16(R4), [V6.S4]
	VFMLA  V8.S4, V6.S4, V3.S4
	VLD1.P 16(R4), [V7.S4]
	VFMLA  V9.S4, V7.S4, V3.S4

	SUB  $1, R7
	CBNZ R7, qeight

qtail4:
	AND $7, R5, R8
	LSR $2, R8, R10
	CBZ R10, qtail1

	VLD1.P 8(R0), [V12.D1]
	VZIP1  V12.H8, V16.H8, V8.H8
	VLD1.P 16(R1), [V4.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	VLD1.P 16(R2), [V5.S4]
	VFMLA  V8.S4, V5.S4, V1.S4
	VLD1.P 16(R3), [V6.S4]
	VFMLA  V8.S4, V6.S4, V2.S4
	VLD1.P 16(R4), [V7.S4]
	VFMLA  V8.S4, V7.S4, V3.S4

qtail1:
	FMOVS $(0.0), F20
	FMOVS $(0.0), F21
	FMOVS $(0.0), F22
	FMOVS $(0.0), F23
	AND   $3, R5, R11
	CBZ   R11, qstore

qone:
	MOVHU.P 2(R0), R12
	LSL     $16, R12
	FMOVS   R12, F19

	FMOVS.P 4(R1), F12
	FMADDS  F19, F20, F12, F20
	FMOVS.P 4(R2), F13
	FMADDS  F19, F21, F13, F21
	FMOVS.P 4(R3), F14
	FMADDS  F19, F22, F14, F22
	FMOVS.P 4(R4), F15
	FMADDS  F19, F23, F15, F23

	SUB  $1, R11
	CBNZ R11, qone

qstore:
	VST1  [V0.S4, V1.S4, V2.S4, V3.S4], (R6)
	FMOVS F20, 64(R6)
	FMOVS F21, 68(R6)
	FMOVS F22, 72(R6)
	FMOVS F23, 76(R6)
	RET
