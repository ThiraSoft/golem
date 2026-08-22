//go:build amd64

#include "textflag.h"

// func scoresF32AVX2(q *float32, k *float32, hd, n int, out *float32)
//
// One query against n keys, the keys laid out one after another: out[j] is the
// dot product of q with the j-th of them.
//
// The loop over the keys is here rather than in Go because it is what the
// attention of a vision tower is made of — a thousand keys, a thousand
// queries, twelve heads, sixteen blocks, and a head is sixty-four floats. A
// dot product that short spends more of itself on the call and on folding four
// accumulators into one number than on multiplying, and doing four keys at a
// time is what amortizes both.
TEXT ·scoresF32AVX2(SB), NOSPLIT, $0-40
	MOVQ q+0(FP), SI
	MOVQ k+8(FP), DI
	MOVQ hd+16(FP), CX
	MOVQ n+24(FP), R11
	MOVQ out+32(FP), R9

	MOVQ CX, R8
	SHLQ $2, R8            // the stride from one key to the next, in bytes

	XORQ R10, R10          // the key being scored

	MOVQ R11, DX
	ANDQ $-4, DX
	JMP  j4cond

j4:
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	// Four more for the tail of a head that is not a multiple of eight. A
	// scalar multiply-accumulate is VEX-encoded and zeroes everything above
	// its own lane, so it may not touch an accumulator a vector one is using.
	VXORPS X12, X12, X12
	VXORPS X13, X13, X13
	VXORPS X14, X14, X14
	VXORPS X15, X15, X15

	// The four keys, addressed from four bases so that one index walks them.
	MOVQ DI, R12
	LEAQ (DI)(R8*1), R13
	LEAQ (DI)(R8*2), R14
	LEAQ (R13)(R8*2), R15

	XORQ AX, AX
	MOVQ CX, BX
	ANDQ $-8, BX
	JMP  d8cond

d8:
	VMOVUPS     (SI)(AX*4), Y8
	VMOVUPS     (R12)(AX*4), Y4
	VFMADD231PS Y4, Y8, Y0
	VMOVUPS     (R13)(AX*4), Y5
	VFMADD231PS Y5, Y8, Y1
	VMOVUPS     (R14)(AX*4), Y6
	VFMADD231PS Y6, Y8, Y2
	VMOVUPS     (R15)(AX*4), Y7
	VFMADD231PS Y7, Y8, Y3
	ADDQ $8, AX

d8cond:
	CMPQ AX, BX
	JLT  d8

	JMP d1cond

d1:
	VMOVSS      (SI)(AX*4), X8
	VMOVSS      (R12)(AX*4), X4
	VFMADD231SS X4, X8, X12
	VMOVSS      (R13)(AX*4), X5
	VFMADD231SS X5, X8, X13
	VMOVSS      (R14)(AX*4), X6
	VFMADD231SS X6, X8, X14
	VMOVSS      (R15)(AX*4), X7
	VFMADD231SS X7, X8, X15
	INCQ        AX

d1cond:
	CMPQ AX, CX
	JLT  d1

	VEXTRACTF128 $1, Y0, X4
	VADDPS       X4, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VADDSS       X12, X0, X0
	VMOVSS       X0, (R9)(R10*4)

	VEXTRACTF128 $1, Y1, X5
	VADDPS       X5, X1, X1
	VHADDPS      X1, X1, X1
	VHADDPS      X1, X1, X1
	VADDSS       X13, X1, X1
	VMOVSS       X1, 4(R9)(R10*4)

	VEXTRACTF128 $1, Y2, X6
	VADDPS       X6, X2, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	VADDSS       X14, X2, X2
	VMOVSS       X2, 8(R9)(R10*4)

	VEXTRACTF128 $1, Y3, X7
	VADDPS       X7, X3, X3
	VHADDPS      X3, X3, X3
	VHADDPS      X3, X3, X3
	VADDSS       X15, X3, X3
	VMOVSS       X3, 12(R9)(R10*4)

	LEAQ (R15)(R8*1), DI   // on to the next four keys
	ADDQ $4, R10

j4cond:
	CMPQ R10, DX
	JLT  j4

	JMP j1cond

j1:
	VXORPS Y0, Y0, Y0
	VXORPS X12, X12, X12
	XORQ   AX, AX
	MOVQ   CX, BX
	ANDQ   $-8, BX
	JMP    e8cond

e8:
	VMOVUPS     (SI)(AX*4), Y8
	VMOVUPS     (DI)(AX*4), Y4
	VFMADD231PS Y4, Y8, Y0
	ADDQ        $8, AX

e8cond:
	CMPQ AX, BX
	JLT  e8

	JMP e1cond

e1:
	VMOVSS      (SI)(AX*4), X8
	VMOVSS      (DI)(AX*4), X4
	VFMADD231SS X4, X8, X12
	INCQ        AX

e1cond:
	CMPQ AX, CX
	JLT  e1

	VEXTRACTF128 $1, Y0, X4
	VADDPS       X4, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VADDSS       X12, X0, X0
	VMOVSS       X0, (R9)(R10*4)

	ADDQ R8, DI
	ADDQ $1, R10

j1cond:
	CMPQ R10, R11
	JLT  j1

	VZEROUPPER
	RET

// func mixF32AVX2(dst *float32, v *float32, w *float32, hd, n int)
//
// dst += sum over j of w[j] * v[j], the value rows laid out one after another.
//
// The whole sum is here for the same reason the scores are: a head is
// sixty-four floats, and a thousand of them accumulated one call at a time
// spends its time on the calls. The accumulator stays in registers for the
// whole sweep, so each value row costs a broadcast, a load and a
// multiply-accumulate and nothing else — where a general axpy would reload and
// store the destination for every one of them.
TEXT ·mixF32AVX2(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), R9
	MOVQ v+8(FP), DI
	MOVQ w+16(FP), R11
	MOVQ hd+24(FP), CX
	MOVQ n+32(FP), R10

	CMPQ CX, $64
	JNE  general

	// The head this tower has: sixty-four floats, which is eight registers,
	// and the sum never leaves them.
	VMOVUPS (R9), Y0
	VMOVUPS 32(R9), Y1
	VMOVUPS 64(R9), Y2
	VMOVUPS 96(R9), Y3
	VMOVUPS 128(R9), Y4
	VMOVUPS 160(R9), Y5
	VMOVUPS 192(R9), Y6
	VMOVUPS 224(R9), Y7

	XORQ AX, AX
	JMP  m64cond

m64:
	VBROADCASTSS (R11)(AX*4), Y8
	VMOVUPS      (DI), Y9
	VFMADD231PS  Y9, Y8, Y0
	VMOVUPS      32(DI), Y10
	VFMADD231PS  Y10, Y8, Y1
	VMOVUPS      64(DI), Y11
	VFMADD231PS  Y11, Y8, Y2
	VMOVUPS      96(DI), Y12
	VFMADD231PS  Y12, Y8, Y3
	VMOVUPS      128(DI), Y9
	VFMADD231PS  Y9, Y8, Y4
	VMOVUPS      160(DI), Y10
	VFMADD231PS  Y10, Y8, Y5
	VMOVUPS      192(DI), Y11
	VFMADD231PS  Y11, Y8, Y6
	VMOVUPS      224(DI), Y12
	VFMADD231PS  Y12, Y8, Y7
	ADDQ         $256, DI
	INCQ         AX

m64cond:
	CMPQ AX, R10
	JLT  m64

	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	VMOVUPS Y2, 64(R9)
	VMOVUPS Y3, 96(R9)
	VMOVUPS Y4, 128(R9)
	VMOVUPS Y5, 160(R9)
	VMOVUPS Y6, 192(R9)
	VMOVUPS Y7, 224(R9)
	VZEROUPPER
	RET

general:
	// Any other width, a row at a time: correct, and never taken here.
	MOVQ CX, R8
	SHLQ $2, R8
	XORQ AX, AX
	JMP  rowcond

row:
	VBROADCASTSS (R11)(AX*4), Y8
	XORQ         BX, BX
	MOVQ         CX, DX
	ANDQ         $-8, DX
	JMP          row8cond

row8:
	VMOVUPS     (R9)(BX*4), Y0
	VMOVUPS     (DI)(BX*4), Y9
	VFMADD231PS Y9, Y8, Y0
	VMOVUPS     Y0, (R9)(BX*4)
	ADDQ        $8, BX

row8cond:
	CMPQ BX, DX
	JLT  row8

	JMP row1cond

row1:
	VMOVSS      (DI)(BX*4), X9
	VMULSS      X8, X9, X9
	VADDSS      (R9)(BX*4), X9, X9
	VMOVSS      X9, (R9)(BX*4)
	INCQ        BX

row1cond:
	CMPQ BX, CX
	JLT  row1

	ADDQ R8, DI
	INCQ AX

rowcond:
	CMPQ AX, R10
	JLT  row

	VZEROUPPER
	RET
