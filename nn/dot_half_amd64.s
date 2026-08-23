//go:build amd64

#include "textflag.h"

// func dotF32HalfAVX2(a *float32, b *uint16, n int) float32
//
// The same shape as dotF32AVX2, deliberately: four accumulators of eight lanes,
// then the eights, then the remainder one at a time, and no fused multiply-add.
//
// The shape is not a style choice. Summing the same products in a different
// order rounds differently, and a mixture of experts routes on a hard top-k of
// its router's logits — a difference of one part in ten million there picks a
// different expert, and the answer diverges for real. This kernel exists so the
// cache can hold halves without moving a single bit of the result.
//
// VCVTPH2PS widens eight halves from sixteen bytes into a full vector, which is
// exactly one accumulator's worth per instruction.
TEXT ·dotF32HalfAVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), SI
	MOVQ b+8(FP), DI
	MOVQ n+16(FP), CX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	XORQ   AX, AX

	MOVQ CX, BX
	SHRQ $5, BX             // BX = groups of 32
	JZ   eights

wide:
	VMOVUPS   (SI)(AX*4), Y4
	VMOVUPS   32(SI)(AX*4), Y5
	VMOVUPS   64(SI)(AX*4), Y6
	VMOVUPS   96(SI)(AX*4), Y7
	VCVTPH2PS (DI)(AX*2), Y8
	VCVTPH2PS 16(DI)(AX*2), Y9
	VCVTPH2PS 32(DI)(AX*2), Y10
	VCVTPH2PS 48(DI)(AX*2), Y11
	VMULPS    Y8, Y4, Y4
	VMULPS    Y9, Y5, Y5
	VMULPS    Y10, Y6, Y6
	VMULPS    Y11, Y7, Y7
	VADDPS    Y4, Y0, Y0
	VADDPS    Y5, Y1, Y1
	VADDPS    Y6, Y2, Y2
	VADDPS    Y7, Y3, Y3
	ADDQ      $32, AX
	DECQ      BX
	JNZ       wide

eights:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

	MOVQ CX, BX
	SUBQ AX, BX
	SHRQ $3, BX
	JZ   reduce

eight:
	VMOVUPS   (SI)(AX*4), Y4
	VCVTPH2PS (DI)(AX*2), Y8
	VMULPS    Y8, Y4, Y4
	VADDPS    Y4, Y0, Y0
	ADDQ      $8, AX
	DECQ      BX
	JNZ       eight

reduce:
	VEXTRACTF128 $1, Y0, X4
	VADDPS       X4, X0, X0
	VMOVHLPS     X0, X0, X4
	VADDPS       X4, X0, X0
	VSHUFPS      $0x01, X0, X0, X4
	VADDSS       X4, X0, X0

tail:
	CMPQ AX, CX
	JGE  done
	MOVWLZX   (DI)(AX*2), BX
	VMOVD     BX, X4
	VCVTPH2PS X4, X4
	VMOVSS    (SI)(AX*4), X5
	VMULSS    X4, X5, X4
	VADDSS    X4, X0, X0
	INCQ      AX
	JMP       tail

done:
	VMOVSS X0, ret+24(FP)
	VZEROUPPER
	RET

// func axpyHalfAVX2(dst *float32, src *uint16, n int, a float32)
//
// axpyAVX2's shape exactly, including the fused multiply-add in the vector path
// and the unfused pair in the tail: this has to give the same bits as widening
// the halves and calling Axpy, or a mixture routes differently.
TEXT ·axpyHalfAVX2(SB), NOSPLIT, $0-28
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS a+24(FP), Y0
	XORQ AX, AX

	MOVQ CX, DX
	ANDQ $-32, DX
	JMP  h32cond

h32:
	VMOVUPS   (DI)(AX*4), Y1
	VMOVUPS   32(DI)(AX*4), Y2
	VMOVUPS   64(DI)(AX*4), Y3
	VMOVUPS   96(DI)(AX*4), Y4
	VCVTPH2PS (SI)(AX*2), Y5
	VCVTPH2PS 16(SI)(AX*2), Y6
	VCVTPH2PS 32(SI)(AX*2), Y7
	VCVTPH2PS 48(SI)(AX*2), Y8
	VFMADD231PS Y5, Y0, Y1
	VFMADD231PS Y6, Y0, Y2
	VFMADD231PS Y7, Y0, Y3
	VFMADD231PS Y8, Y0, Y4
	VMOVUPS Y1, (DI)(AX*4)
	VMOVUPS Y2, 32(DI)(AX*4)
	VMOVUPS Y3, 64(DI)(AX*4)
	VMOVUPS Y4, 96(DI)(AX*4)
	ADDQ    $32, AX

h32cond:
	CMPQ AX, DX
	JLT  h32

	MOVQ CX, DX
	ANDQ $-8, DX
	JMP  h8cond

h8:
	VMOVUPS     (DI)(AX*4), Y1
	VCVTPH2PS   (SI)(AX*2), Y5
	VFMADD231PS Y5, Y0, Y1
	VMOVUPS     Y1, (DI)(AX*4)
	ADDQ        $8, AX

h8cond:
	CMPQ AX, DX
	JLT  h8

	JMP h1cond

h1:
	MOVWLZX   (SI)(AX*2), BX
	VMOVD     BX, X5
	VCVTPH2PS X5, X5
	VMULSS    X0, X5, X5
	VADDSS    (DI)(AX*4), X5, X5
	VMOVSS    X5, (DI)(AX*4)
	INCQ      AX

h1cond:
	CMPQ AX, CX
	JLT  h1

	VZEROUPPER
	RET
