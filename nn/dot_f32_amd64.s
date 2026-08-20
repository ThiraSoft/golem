//go:build amd64

#include "textflag.h"

// func dotF32AVX2(a, b *float32, n int) float32
//
// Four accumulators of eight lanes, then the eights, then the remainder one at
// a time. Nothing is fused: an FMA would round once where the reference rounds
// twice, and these products feed a comparison against llama.cpp.
TEXT ·dotF32AVX2(SB), NOSPLIT, $0-28
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
	VMOVUPS (SI)(AX*4), Y4
	VMOVUPS 32(SI)(AX*4), Y5
	VMOVUPS 64(SI)(AX*4), Y6
	VMOVUPS 96(SI)(AX*4), Y7
	VMULPS  (DI)(AX*4), Y4, Y4
	VMULPS  32(DI)(AX*4), Y5, Y5
	VMULPS  64(DI)(AX*4), Y6, Y6
	VMULPS  96(DI)(AX*4), Y7, Y7
	VADDPS  Y4, Y0, Y0
	VADDPS  Y5, Y1, Y1
	VADDPS  Y6, Y2, Y2
	VADDPS  Y7, Y3, Y3
	ADDQ    $32, AX
	DECQ    BX
	JNZ     wide

eights:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

	MOVQ CX, BX
	SUBQ AX, BX
	SHRQ $3, BX
	JZ   reduce

eight:
	VMOVUPS (SI)(AX*4), Y4
	VMULPS  (DI)(AX*4), Y4, Y4
	VADDPS  Y4, Y0, Y0
	ADDQ    $8, AX
	DECQ    BX
	JNZ     eight

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
	VMOVSS (SI)(AX*4), X4
	VMULSS (DI)(AX*4), X4, X4
	VADDSS X4, X0, X0
	INCQ   AX
	JMP    tail

done:
	VMOVSS X0, ret+24(FP)
	VZEROUPPER
	RET
