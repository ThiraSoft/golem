//go:build amd64

#include "textflag.h"

// func dotBF16x2x4AVX2(row *uint16, rowStride int, x *float32, colStride, n int, out *float32)
//
// Two rows of bfloat16 weights against four float32 vectors: eight dot
// products in one sweep, written to out as row-major two by four.
//
// This is the shape that stops the loads from being the limit. One row against
// one column spends ten loads on eight multiply-accumulates, and this machine
// issues two loads a cycle against two multiply-accumulates: the loads finish
// last. Blocking in both directions makes every activation feed two rows and
// every row feed four activations — six loads for eight — and the arithmetic
// becomes what the kernel waits for, which is the most a kernel can hope for.
//
// n must be a multiple of eight; the caller has a narrower kernel for what is
// left over.
TEXT ·dotBF16x2x4AVX2(SB), NOSPLIT, $0-48
	MOVQ row+0(FP), SI
	MOVQ rowStride+8(FP), R11
	MOVQ x+16(FP), DI
	MOVQ colStride+24(FP), R8
	MOVQ n+32(FP), CX
	MOVQ out+40(FP), R9

	SHLQ $1, R11           // the second row, in bytes
	ADDQ SI, R11
	SHLQ $2, R8            // the stride from one column to the next, in bytes

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	MOVQ DI, R12
	LEAQ (DI)(R8*1), R13
	LEAQ (DI)(R8*2), R14
	LEAQ (R13)(R8*2), R15

	XORQ AX, AX
	JMP  c8cond

c8:
	PREFETCHT0 2048(SI)(AX*2)
	PREFETCHT0 2048(R11)(AX*2)
	VPMOVZXWD (SI)(AX*2), Y8
	VPSLLD    $16, Y8, Y8
	VPMOVZXWD (R11)(AX*2), Y9
	VPSLLD    $16, Y9, Y9

	VMOVUPS     (R12)(AX*4), Y10
	VFMADD231PS Y10, Y8, Y0
	VFMADD231PS Y10, Y9, Y4

	VMOVUPS     (R13)(AX*4), Y11
	VFMADD231PS Y11, Y8, Y1
	VFMADD231PS Y11, Y9, Y5

	VMOVUPS     (R14)(AX*4), Y12
	VFMADD231PS Y12, Y8, Y2
	VFMADD231PS Y12, Y9, Y6

	VMOVUPS     (R15)(AX*4), Y13
	VFMADD231PS Y13, Y8, Y3
	VFMADD231PS Y13, Y9, Y7

	ADDQ $8, AX

c8cond:
	CMPQ AX, CX
	JLT  c8

	// Eight horizontal sums, the first row then the second.
	VEXTRACTF128 $1, Y0, X8
	VADDPS       X8, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VMOVSS       X0, (R9)

	VEXTRACTF128 $1, Y1, X9
	VADDPS       X9, X1, X1
	VHADDPS      X1, X1, X1
	VHADDPS      X1, X1, X1
	VMOVSS       X1, 4(R9)

	VEXTRACTF128 $1, Y2, X10
	VADDPS       X10, X2, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	VMOVSS       X2, 8(R9)

	VEXTRACTF128 $1, Y3, X11
	VADDPS       X11, X3, X3
	VHADDPS      X3, X3, X3
	VHADDPS      X3, X3, X3
	VMOVSS       X3, 12(R9)

	VEXTRACTF128 $1, Y4, X8
	VADDPS       X8, X4, X4
	VHADDPS      X4, X4, X4
	VHADDPS      X4, X4, X4
	VMOVSS       X4, 16(R9)

	VEXTRACTF128 $1, Y5, X9
	VADDPS       X9, X5, X5
	VHADDPS      X5, X5, X5
	VHADDPS      X5, X5, X5
	VMOVSS       X5, 20(R9)

	VEXTRACTF128 $1, Y6, X10
	VADDPS       X10, X6, X6
	VHADDPS      X6, X6, X6
	VHADDPS      X6, X6, X6
	VMOVSS       X6, 24(R9)

	VEXTRACTF128 $1, Y7, X11
	VADDPS       X11, X7, X7
	VHADDPS      X7, X7, X7
	VHADDPS      X7, X7, X7
	VMOVSS       X7, 28(R9)

	VZEROUPPER
	RET
