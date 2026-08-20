//go:build amd64

#include "textflag.h"

// func dotBF16AVX2(row *uint16, x *float32, n int) float32
//
// Dot product of a row of bfloat16 weights with a float32 vector.
// The bfloat16 -> float32 conversion is a sixteen-bit shift; VPMOVZXWD widens
// eight weights at once, VPSLLD shifts them, VFMADD231PS accumulates. Eight
// multiply-accumulates per instruction, against one in Go.
TEXT ·dotBF16AVX2(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ n+16(FP), CX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS X8, X8, X8
	XORQ   AX, AX

	// Four independent accumulators: FMA latency is four cycles, a single
	// accumulator would leave the execution unit idle.
	MOVQ CX, DX
	ANDQ $-32, DX
	JMP  b32cond

b32:
	// Prefetching matters: the transformer's matrices are two to eight
	// megabytes, spread over eight cores, and the hardware prefetcher does not
	// have time to get up to speed before the row ends.
	PREFETCHT0 2048(SI)(AX*2)
	VPMOVZXWD (SI)(AX*2), Y4
	VPSLLD    $16, Y4, Y4
	VMOVUPS   (DI)(AX*4), Y12
	VFMADD231PS Y12, Y4, Y0

	VPMOVZXWD 16(SI)(AX*2), Y5
	VPSLLD    $16, Y5, Y5
	VMOVUPS   32(DI)(AX*4), Y13
	VFMADD231PS Y13, Y5, Y1

	VPMOVZXWD 32(SI)(AX*2), Y6
	VPSLLD    $16, Y6, Y6
	VMOVUPS   64(DI)(AX*4), Y14
	VFMADD231PS Y14, Y6, Y2

	VPMOVZXWD 48(SI)(AX*2), Y7
	VPSLLD    $16, Y7, Y7
	VMOVUPS   96(DI)(AX*4), Y15
	VFMADD231PS Y15, Y7, Y3

	ADDQ $32, AX

b32cond:
	CMPQ AX, DX
	JLT  b32

	// Remainder: by eights, then one by one.
	MOVQ CX, DX
	ANDQ $-8, DX
	JMP  b8cond

b8:
	VPMOVZXWD (SI)(AX*2), Y4
	VPSLLD    $16, Y4, Y4
	VMOVUPS   (DI)(AX*4), Y12
	VFMADD231PS Y12, Y4, Y0
	ADDQ $8, AX

b8cond:
	CMPQ AX, DX
	JLT  b8

	JMP b1cond

b1:
	MOVWLZX (SI)(AX*2), BX
	SHLL    $16, BX
	VMOVD   BX, X4
	VMULSS  (DI)(AX*4), X4, X4
	VADDSS  X4, X8, X8
	INCQ    AX

b1cond:
	CMPQ AX, CX
	JLT  b1

	// Horizontal reduction.
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VADDSS  X8, X0, X0
	VMOVSS  X0, ret+24(FP)
	VZEROUPPER
	RET

// func hasAVX2() bool
TEXT ·hasAVX2(SB), NOSPLIT, $0-1
	MOVQ $0, AX
	CPUID
	CMPQ AX, $7
	JLT  no

	MOVL $1, AX
	MOVL $0, CX
	CPUID
	ANDL $(1<<12), CX  // FMA
	JZ   no

	MOVL $7, AX
	MOVL $0, CX
	CPUID
	ANDL $(1<<5), BX   // AVX2
	JZ   no

	MOVB $1, ret+0(FP)
	RET

no:
	MOVB $0, ret+0(FP)
	RET

// func axpyAVX2(dst, src *float32, n int, a float32)
//
// dst += a*src. This is the inner loop of the convolutions: one fixed
// coefficient, swept along the output. Go does not vectorize; eight positions
// per FMA change the regime.
TEXT ·axpyAVX2(SB), NOSPLIT, $0-28
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS a+24(FP), Y0
	XORQ AX, AX

	MOVQ CX, DX
	ANDQ $-32, DX
	JMP  a32cond

a32:
	VMOVUPS (DI)(AX*4), Y1
	VMOVUPS 32(DI)(AX*4), Y2
	VMOVUPS 64(DI)(AX*4), Y3
	VMOVUPS 96(DI)(AX*4), Y4
	VMOVUPS (SI)(AX*4), Y5
	VMOVUPS 32(SI)(AX*4), Y6
	VMOVUPS 64(SI)(AX*4), Y7
	VMOVUPS 96(SI)(AX*4), Y8
	VFMADD231PS Y5, Y0, Y1
	VFMADD231PS Y6, Y0, Y2
	VFMADD231PS Y7, Y0, Y3
	VFMADD231PS Y8, Y0, Y4
	VMOVUPS Y1, (DI)(AX*4)
	VMOVUPS Y2, 32(DI)(AX*4)
	VMOVUPS Y3, 64(DI)(AX*4)
	VMOVUPS Y4, 96(DI)(AX*4)
	ADDQ $32, AX

a32cond:
	CMPQ AX, DX
	JLT  a32

	MOVQ CX, DX
	ANDQ $-8, DX
	JMP  a8cond

a8:
	VMOVUPS (DI)(AX*4), Y1
	VMOVUPS (SI)(AX*4), Y5
	VFMADD231PS Y5, Y0, Y1
	VMOVUPS Y1, (DI)(AX*4)
	ADDQ $8, AX

a8cond:
	CMPQ AX, DX
	JLT  a8

	JMP a1cond

a1:
	VMOVSS (SI)(AX*4), X5
	VMULSS X0, X5, X5
	VADDSS (DI)(AX*4), X5, X5
	VMOVSS X5, (DI)(AX*4)
	INCQ   AX

a1cond:
	CMPQ AX, CX
	JLT  a1

	VZEROUPPER
	RET
