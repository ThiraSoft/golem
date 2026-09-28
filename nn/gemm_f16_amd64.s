//go:build amd64

#include "textflag.h"

// STEP is one k of the tile: sixteen weights widened into Y12 and Y13, then
// each of the six columns' value at k broadcast and met by both.
#define STEP(off) \
	VCVTPH2PS off(SI), Y12 \
	VCVTPH2PS off+16(SI), Y13 \
	VBROADCASTSS (BX), Y14 \
	VFMADD231PS Y12, Y14, Y0 \
	VFMADD231PS Y13, Y14, Y1 \
	VBROADCASTSS (BX)(R8*1), Y15 \
	VFMADD231PS Y12, Y15, Y2 \
	VFMADD231PS Y13, Y15, Y3 \
	VBROADCASTSS (BX)(R8*2), Y14 \
	VFMADD231PS Y12, Y14, Y4 \
	VFMADD231PS Y13, Y14, Y5 \
	VBROADCASTSS (R10), Y15 \
	VFMADD231PS Y12, Y15, Y6 \
	VFMADD231PS Y13, Y15, Y7 \
	VBROADCASTSS (R10)(R8*1), Y14 \
	VFMADD231PS Y12, Y14, Y8 \
	VFMADD231PS Y13, Y14, Y9 \
	VBROADCASTSS (R10)(R8*2), Y15 \
	VFMADD231PS Y12, Y15, Y10 \
	VFMADD231PS Y13, Y15, Y11

// ADDC adds two accumulators into sixteen floats of C at DX and moves DX on
// to the next column.
#define ADDC(A, B) \
	VADDPS (DX), A, A \
	VADDPS 32(DX), B, B \
	VMOVUPS A, (DX) \
	VMOVUPS B, 32(DX) \
	ADDQ R9, DX

// func gemmF16x16x6AVX2(w *uint16, x *float32, xStride, k int, c *float32, cStride int)
//
// Sixteen outputs by six columns over k of the shared dimension, k even,
// added into C. w is a packed panel, sixteen halves a k; column j of the
// operand starts xStride*j bytes after x, and column j of C cStride*j bytes
// after c, sixteen floats.
//
// Each weight is widened once and meets six columns, and each column's value
// is broadcast once and meets sixteen weights: twelve accumulators, two
// widened weights and a broadcast are fifteen of the sixteen registers, and
// the loads are eight to twelve multiply-adds, which leaves the multiply-adds
// the bound.
TEXT ·gemmF16x16x6AVX2(SB), NOSPLIT, $0-48
	MOVQ w+0(FP), SI
	MOVQ x+8(FP), BX
	MOVQ xStride+16(FP), R8
	MOVQ k+24(FP), CX
	MOVQ c+32(FP), DX
	MOVQ cStride+40(FP), R9
	LEAQ (BX)(R8*2), R10
	ADDQ R8, R10
	SHRQ $1, CX

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
	JZ    store

loop:
	STEP(0)
	ADDQ $4, BX
	ADDQ $4, R10
	STEP(32)
	ADDQ $4, BX
	ADDQ $4, R10
	ADDQ $64, SI
	DECQ CX
	JNZ  loop

store:
	ADDC(Y0, Y1)
	ADDC(Y2, Y3)
	ADDC(Y4, Y5)
	ADDC(Y6, Y7)
	ADDC(Y8, Y9)
	ADDC(Y10, Y11)
	VZEROUPPER
	RET

// STEP32 is one k of the float32 tile: sixteen values of a loaded, then as
// STEP.
#define STEP32 \
	VMOVUPS (SI), Y12 \
	VMOVUPS 32(SI), Y13 \
	VBROADCASTSS (BX), Y14 \
	VFMADD231PS Y12, Y14, Y0 \
	VFMADD231PS Y13, Y14, Y1 \
	VBROADCASTSS (BX)(R8*1), Y15 \
	VFMADD231PS Y12, Y15, Y2 \
	VFMADD231PS Y13, Y15, Y3 \
	VBROADCASTSS (BX)(R8*2), Y14 \
	VFMADD231PS Y12, Y14, Y4 \
	VFMADD231PS Y13, Y14, Y5 \
	VBROADCASTSS (R10), Y15 \
	VFMADD231PS Y12, Y15, Y6 \
	VFMADD231PS Y13, Y15, Y7 \
	VBROADCASTSS (R10)(R8*1), Y14 \
	VFMADD231PS Y12, Y14, Y8 \
	VFMADD231PS Y13, Y14, Y9 \
	VBROADCASTSS (R10)(R8*2), Y15 \
	VFMADD231PS Y12, Y15, Y10 \
	VFMADD231PS Y13, Y15, Y11

// PUTC stores two accumulators as sixteen floats of C at DX and moves DX on
// to the next column.
#define PUTC(A, B) \
	VMOVUPS A, (DX) \
	VMOVUPS B, 32(DX) \
	ADDQ R9, DX

// func gemmF32x16x6AVX2(a *float32, aStride int, x *float32, xStride, k int, c *float32, cStride int)
//
// gemmF16x16x6AVX2 with float32 rows of sixteen, aStride bytes apart, and
// the sums stored into C rather than added: sixteen outputs by six columns
// over k of the shared dimension, any k.
TEXT ·gemmF32x16x6AVX2(SB), NOSPLIT, $0-56
	MOVQ a+0(FP), SI
	MOVQ aStride+8(FP), AX
	MOVQ x+16(FP), BX
	MOVQ xStride+24(FP), R8
	MOVQ k+32(FP), CX
	MOVQ c+40(FP), DX
	MOVQ cStride+48(FP), R9
	LEAQ (BX)(R8*2), R10
	ADDQ R8, R10

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
	JZ    put

loop32:
	STEP32
	ADDQ AX, SI
	ADDQ $4, BX
	ADDQ $4, R10
	DECQ CX
	JNZ  loop32

put:
	PUTC(Y0, Y1)
	PUTC(Y2, Y3)
	PUTC(Y4, Y5)
	PUTC(Y6, Y7)
	PUTC(Y8, Y9)
	PUTC(Y10, Y11)
	VZEROUPPER
	RET
