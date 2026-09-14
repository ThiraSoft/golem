//go:build amd64

#include "textflag.h"

// REDUCE sums the eight lanes of an accumulator into its lowest one, in the
// order dotF32HalfAVX2 reduces, and stores it. X12 is free by then.
#define REDUCE(Y, X, off) \
	VEXTRACTF128 $1, Y, X12 \
	VADDPS       X12, X, X \
	VMOVHLPS     X, X, X12 \
	VADDPS       X12, X, X \
	VSHUFPS      $0x01, X, X, X12 \
	VADDSS       X12, X, X \
	VMOVSS       X, off(DX)

// func matF16x4x3AVX2(w *uint16, wStride int, x *float32, xStride, n int, out *float32)
//
// Twelve dot products at once: four fp16 weight rows against three float32
// activation columns, n long, n a multiple of eight. out gets them row by row,
// out[r*3+c].
//
// The point is reuse. dotF32HalfAVX2 widens a row of halves for every column
// it meets and reads it again for the next; here a row is widened once per
// eight columns of the step and meets three columns while it sits in a
// register. Twelve accumulators and four widened rows are the sixteen ymm
// registers there are, so the columns are read as the memory operand of the
// fused multiply-add rather than loaded.
//
// It is not bit-compatible with dotF32HalfAVX2 and does not try to be: it
// fuses, and it keeps one accumulator a product where that one keeps four.
// The only engine that calls it is an encoder whose reference, ggml's fp16
// product, is neither.
TEXT ·matF16x4x3AVX2(SB), NOSPLIT, $0-48
	MOVQ w+0(FP), SI
	MOVQ wStride+8(FP), R8
	SHLQ $1, R8                 // bytes between rows
	MOVQ x+16(FP), BX
	MOVQ xStride+24(FP), R11
	SHLQ $2, R11                // bytes between columns
	MOVQ n+32(FP), CX
	SHRQ $3, CX                 // steps of eight
	MOVQ out+40(FP), DX
	LEAQ (SI)(R8*1), R10        // row 1; rows 2 and 3 are R10+R8 and R10+2*R8

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11

	TESTQ CX, CX
	JZ    reduce

step:
	VCVTPH2PS (SI), Y12
	VCVTPH2PS (R10), Y13
	VCVTPH2PS (R10)(R8*1), Y14
	VCVTPH2PS (R10)(R8*2), Y15

	VFMADD231PS (BX), Y12, Y0
	VFMADD231PS (BX)(R11*1), Y12, Y1
	VFMADD231PS (BX)(R11*2), Y12, Y2
	VFMADD231PS (BX), Y13, Y3
	VFMADD231PS (BX)(R11*1), Y13, Y4
	VFMADD231PS (BX)(R11*2), Y13, Y5
	VFMADD231PS (BX), Y14, Y6
	VFMADD231PS (BX)(R11*1), Y14, Y7
	VFMADD231PS (BX)(R11*2), Y14, Y8
	VFMADD231PS (BX), Y15, Y9
	VFMADD231PS (BX)(R11*1), Y15, Y10
	VFMADD231PS (BX)(R11*2), Y15, Y11

	ADDQ $16, SI
	ADDQ $16, R10
	ADDQ $32, BX
	DECQ CX
	JNZ  step

reduce:
	REDUCE(Y0, X0, 0)
	REDUCE(Y1, X1, 4)
	REDUCE(Y2, X2, 8)
	REDUCE(Y3, X3, 12)
	REDUCE(Y4, X4, 16)
	REDUCE(Y5, X5, 20)
	REDUCE(Y6, X6, 24)
	REDUCE(Y7, X7, 28)
	REDUCE(Y8, X8, 32)
	REDUCE(Y9, X9, 36)
	REDUCE(Y10, X10, 40)
	REDUCE(Y11, X11, 44)
	VZEROUPPER
	RET

// func matF32x4x3AVX2(w *float32, wStride int, x *float32, xStride, n int, out *float32)
//
// matF16x4x3AVX2 with float32 rows: the rows are loaded rather than widened,
// and everything else is the same. It is what an encoder's attention runs on —
// queries against keys, then probabilities against transposed values.
TEXT ·matF32x4x3AVX2(SB), NOSPLIT, $0-48
	MOVQ w+0(FP), SI
	MOVQ wStride+8(FP), R8
	SHLQ $2, R8                 // bytes between rows
	MOVQ x+16(FP), BX
	MOVQ xStride+24(FP), R11
	SHLQ $2, R11                // bytes between columns
	MOVQ n+32(FP), CX
	SHRQ $3, CX                 // steps of eight
	MOVQ out+40(FP), DX
	LEAQ (SI)(R8*1), R10        // row 1; rows 2 and 3 are R10+R8 and R10+2*R8

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11

	TESTQ CX, CX
	JZ    reduce32

step32:
	VMOVUPS (SI), Y12
	VMOVUPS (R10), Y13
	VMOVUPS (R10)(R8*1), Y14
	VMOVUPS (R10)(R8*2), Y15

	VFMADD231PS (BX), Y12, Y0
	VFMADD231PS (BX)(R11*1), Y12, Y1
	VFMADD231PS (BX)(R11*2), Y12, Y2
	VFMADD231PS (BX), Y13, Y3
	VFMADD231PS (BX)(R11*1), Y13, Y4
	VFMADD231PS (BX)(R11*2), Y13, Y5
	VFMADD231PS (BX), Y14, Y6
	VFMADD231PS (BX)(R11*1), Y14, Y7
	VFMADD231PS (BX)(R11*2), Y14, Y8
	VFMADD231PS (BX), Y15, Y9
	VFMADD231PS (BX)(R11*1), Y15, Y10
	VFMADD231PS (BX)(R11*2), Y15, Y11

	ADDQ $32, SI
	ADDQ $32, R10
	ADDQ $32, BX
	DECQ CX
	JNZ  step32

reduce32:
	REDUCE(Y0, X0, 0)
	REDUCE(Y1, X1, 4)
	REDUCE(Y2, X2, 8)
	REDUCE(Y3, X3, 12)
	REDUCE(Y4, X4, 16)
	REDUCE(Y5, X5, 20)
	REDUCE(Y6, X6, 24)
	REDUCE(Y7, X7, 28)
	REDUCE(Y8, X8, 32)
	REDUCE(Y9, X9, 36)
	REDUCE(Y10, X10, 40)
	REDUCE(Y11, X11, 44)
	VZEROUPPER
	RET
