//go:build amd64

#include "textflag.h"

// func roundHalfAVX2(v *float32, n int)
//
// Every attention probability is rounded to fp16 before it multiplies a value,
// because ggml's kernel takes them that way and a score that keeps its float32
// mantissa is not more accurate than the reference, it is different from it.
// The scalar RoundHalf does that with two branchy conversions per number, and
// the loop runs over every visible position of every head of every token — at
// four thousand positions of context it was three and a half percent of a token.
//
// VCVTPS2PH is the same rounding in one instruction for eight numbers: mode 0
// is round-to-nearest-even, which is what floatToHalf implements by hand. The
// results are bit-identical, which matters here for the same reason it mattered
// for the cache — a mixture of experts routes on a hard top-k.
TEXT ·roundHalfAVX2(SB), NOSPLIT, $0-16
	MOVQ v+0(FP), SI
	MOVQ n+8(FP), CX
	XORQ AX, AX

	MOVQ CX, DX
	ANDQ $-8, DX
	JMP  eightcond

eight:
	VMOVUPS   (SI)(AX*4), Y0
	VCVTPS2PH $0, Y0, X1
	VCVTPH2PS X1, Y0
	VMOVUPS   Y0, (SI)(AX*4)
	ADDQ      $8, AX

eightcond:
	CMPQ AX, DX
	JLT  eight

	JMP onecond

one:
	VMOVSS    (SI)(AX*4), X0
	VCVTPS2PH $0, X0, X1
	VCVTPH2PS X1, X0
	VMOVSS    X0, (SI)(AX*4)
	INCQ      AX

onecond:
	CMPQ AX, CX
	JLT  one

	VZEROUPPER
	RET
