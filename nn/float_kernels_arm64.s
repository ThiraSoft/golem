//go:build arm64

#include "textflag.h"

// The two float32 kernels the attention loop is made of.
//
// Scores is a query against every key in the cache, and Mix is the weighted sum
// of the values: DotF32 and Axpy, called once per position per head. Neither
// needs anything Go's assembler will not spell — VFMLA is a fused multiply-add
// on four lanes, which is the whole instruction set this requires.
//
// Four accumulators, as in the AVX2 kernels and for the same reason: at one
// position per token these loops are short enough that what limits them is the
// dependency between the additions rather than the arithmetic. Sixteen elements
// a turn keeps four chains going.
//
// The dot product does not reduce its accumulators here. It writes all four
// out and lets Go fold them, which costs one store of sixteen floats and saves
// hand-encoding a vector FADD the assembler does not know.

// func dotF32LanesNEON(a, b *float32, n int, out *float32)
//
// out receives sixteen partial sums: the four accumulators, each of four lanes.
// Lane j of every one of them holds elements at i ≡ j (mod 4), which is the
// portable loop's s[j], so Go adds the four together lane by lane.
TEXT ·dotF32LanesNEON(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	MOVD out+24(FP), R3

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	// Sixteen at a time, four accumulators.
	MOVD R2, R4
	LSR  $4, R4
	CBZ  R4, tail4

wide:
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	VFMLA  V9.S4, V5.S4, V1.S4
	VFMLA  V10.S4, V6.S4, V2.S4
	VFMLA  V11.S4, V7.S4, V3.S4
	SUB    $1, R4
	CBNZ   R4, wide

tail4:
	// What sixteen did not cover, four at a time, into the first accumulator.
	AND $15, R2, R5
	LSR $2, R5, R6
	CBZ R6, tail1

four:
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R1), [V8.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	SUB    $1, R6
	CBNZ   R6, four

tail1:
	// And the last three or fewer, one at a time, in a scalar of their own —
	// reaching into a lane of an accumulator costs more instructions here than
	// giving the remainder its own slot for Go to add.
	FMOVS $(0.0), F14
	AND   $3, R2, R7
	CBZ   R7, store

one:
	FMOVS.P 4(R0), F12
	FMOVS.P 4(R1), F13
	FMADDS  F13, F14, F12, F14
	SUB     $1, R7
	CBNZ    R7, one

store:
	// All four accumulators go out, then the remainder. Go adds them, because
	// the assembler has no vector float add — VADD is the integer one — and
	// folding them here would mean hand-encoding FADD at the end of a loop.
	VST1  [V0.S4, V1.S4, V2.S4, V3.S4], (R3)
	FMOVS F14, 64(R3)
	RET

// func axpyNEON(dst, src *float32, n int, a float32)
TEXT ·axpyNEON(SB), NOSPLIT, $0-28
	MOVD  dst+0(FP), R0
	MOVD  src+8(FP), R1
	MOVD  n+16(FP), R2
	FMOVS a+24(FP), F0
	VDUP  V0.S[0], V1.S4

	MOVD R2, R4
	LSR  $4, R4
	CBZ  R4, atail4

awide:
	VLD1 (R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VFMLA V1.S4, V8.S4, V4.S4
	VFMLA V1.S4, V9.S4, V5.S4
	VFMLA V1.S4, V10.S4, V6.S4
	VFMLA V1.S4, V11.S4, V7.S4
	VST1.P [V4.S4, V5.S4, V6.S4, V7.S4], 64(R0)
	SUB   $1, R4
	CBNZ  R4, awide

atail4:
	AND $15, R2, R5
	LSR $2, R5, R6
	CBZ R6, atail1

afour:
	VLD1   (R0), [V4.S4]
	VLD1.P 16(R1), [V8.S4]
	VFMLA  V1.S4, V8.S4, V4.S4
	VST1.P [V4.S4], 16(R0)
	SUB    $1, R6
	CBNZ   R6, afour

atail1:
	AND $3, R2, R7
	CBZ R7, adone

aone:
	FMOVS  (R0), F12
	FMOVS.P 4(R1), F13
	FMADDS F13, F12, F0, F12
	FMOVS.P F12, 4(R0)
	SUB    $1, R7
	CBNZ   R7, aone

adone:
	RET
