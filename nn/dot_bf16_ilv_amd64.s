//go:build amd64

#include "textflag.h"

// func dotBF16x2x4IlvAVX2(row *uint16, rowStride int, x *float32, colStride, n int, out *float32)
//
// The same two rows of weights against four columns as the kernel beside it,
// with the widening done by an interleave rather than by a shift.
//
// A bfloat16 becomes a float32 by moving into the high half of a word pair,
// which is what interleaving it with zeros does — and an interleave runs on
// the shuffle port, where a shift runs on the two ports the
// multiply-accumulate itself needs. Those two shifts per eight products were a
// fifth of the ports the arithmetic was waiting for.
//
// What it costs is the order. Interleaving sixteen weights gives the first
// four and the ninth to twelfth in one register and the rest in the other, so
// the activation has to be stored the same way round. That costs nothing: it
// is written out once anyway, clamped and rounded, before any of this runs,
// and Interleave writes it in this order instead of the plain one.
//
// n must be a multiple of sixteen.
TEXT ·dotBF16x2x4IlvAVX2(SB), NOSPLIT, $0-48
	MOVQ row+0(FP), SI
	MOVQ rowStride+8(FP), R11
	MOVQ x+16(FP), DI
	MOVQ colStride+24(FP), R8
	MOVQ n+32(FP), CX
	MOVQ out+40(FP), R9

	SHLQ $1, R11
	ADDQ SI, R11           // the second row
	SHLQ $2, R8            // between two columns, in bytes

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y15, Y15, Y15   // the zeros the weights are interleaved with

	MOVQ DI, R12
	LEAQ (DI)(R8*1), R13
	LEAQ (DI)(R8*2), R14
	LEAQ (R13)(R8*2), R15

	XORQ AX, AX

c16:
	VMOVUPS    (SI)(AX*2), Y14
	VPUNPCKLWD Y14, Y15, Y8
	VPUNPCKHWD Y14, Y15, Y9
	VMOVUPS    (R11)(AX*2), Y14
	VPUNPCKLWD Y14, Y15, Y10
	VPUNPCKHWD Y14, Y15, Y11

	VMOVUPS     (R12)(AX*4), Y12
	VMOVUPS     32(R12)(AX*4), Y13
	VFMADD231PS Y12, Y8, Y0
	VFMADD231PS Y12, Y10, Y4
	VFMADD231PS Y13, Y9, Y0
	VFMADD231PS Y13, Y11, Y4

	VMOVUPS     (R13)(AX*4), Y12
	VMOVUPS     32(R13)(AX*4), Y13
	VFMADD231PS Y12, Y8, Y1
	VFMADD231PS Y12, Y10, Y5
	VFMADD231PS Y13, Y9, Y1
	VFMADD231PS Y13, Y11, Y5

	VMOVUPS     (R14)(AX*4), Y12
	VMOVUPS     32(R14)(AX*4), Y13
	VFMADD231PS Y12, Y8, Y2
	VFMADD231PS Y12, Y10, Y6
	VFMADD231PS Y13, Y9, Y2
	VFMADD231PS Y13, Y11, Y6

	VMOVUPS     (R15)(AX*4), Y12
	VMOVUPS     32(R15)(AX*4), Y13
	VFMADD231PS Y12, Y8, Y3
	VFMADD231PS Y12, Y10, Y7
	VFMADD231PS Y13, Y9, Y3
	VFMADD231PS Y13, Y11, Y7

	ADDQ $16, AX
	CMPQ AX, CX
	JLT  c16

	// Eight horizontal sums: the first row's four, then the second's.
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
