//go:build arm64

#include "textflag.h"

// Rounding a stretch of float32 through fp16 and back.
//
// It is what writes the key-value cache in gemma/attention.go and
// qwen/attention.go, so it runs once per position per head on the generation
// path. Two instructions do it four lanes at a time — narrow to half, widen
// back — and both are hand-encoded, because Go's assembler has FCVTHS and
// FCVTSH for one scalar and nothing for a vector.
//
// FCVTN rounds to nearest even, which is what floatToHalf does. The test beside
// this file holds the two to exact equality over the awkward values as well as
// random ones: subnormals, the overflow edge, infinities and NaN.

// func roundHalfNEON(v *float32, n int)
TEXT ·roundHalfNEON(SB), NOSPLIT, $0-16
	MOVD v+0(FP), R0
	MOVD n+8(FP), R1

	LSR  $2, R1, R2
	CBZ  R2, rtail

rfour:
	VLD1 (R0), [V4.S4]
	WORD $0x0e216888               // FCVTN V8.4H, V4.4S
	WORD $0x0e217904               // FCVTL V4.4S, V8.4H
	VST1.P [V4.S4], 16(R0)
	SUB  $1, R2
	CBNZ R2, rfour

rtail:
	AND $3, R1, R3
	CBZ R3, rdone

rone:
	FMOVS  (R0), F0
	FCVTSH F0, F0
	FCVTHS F0, F0
	FMOVS.P F0, 4(R0)
	SUB    $1, R3
	CBNZ   R3, rone

rdone:
	RET
