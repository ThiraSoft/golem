//go:build amd64

#include "textflag.h"

// func scores4AVX2(q *float32, k *float32, hd, n int, out *float32, outStride int)
//
// Four queries against n keys: four rows of scores, outStride floats apart.
//
// Four rather than one for the same reason the weight kernels take four
// columns at a time. Scoring one query against every key reads the whole of a
// head's keys — a quarter of a megabyte here — and doing that once per query
// reads it a thousand times over. Four queries at a time read it a quarter as
// often, and every key that is loaded feeds four multiply-accumulates instead
// of one.
//
// hd must be a multiple of eight, which sixty-four is.
TEXT ·scores4AVX2(SB), NOSPLIT, $0-48
	MOVQ q+0(FP), SI
	MOVQ k+8(FP), DI
	MOVQ hd+16(FP), CX
	MOVQ n+24(FP), R11
	MOVQ out+32(FP), R9
	MOVQ outStride+40(FP), R10

	SHLQ $2, R10           // between two rows of scores, in bytes
	MOVQ CX, R8
	SHLQ $2, R8            // one head, in bytes

	// The four queries, addressed from four bases.
	MOVQ SI, R12
	LEAQ (SI)(R8*1), R13
	LEAQ (SI)(R8*2), R14
	LEAQ (R13)(R8*2), R15

	XORQ DX, DX            // the key being scored
	MOVQ R11, BX
	ANDQ $-2, BX
	JMP  k2cond

k2:
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	LEAQ (DI)(R8*1), SI    // the second key of the pair
	XORQ AX, AX

d8:
	VMOVUPS (DI)(AX*4), Y12
	VMOVUPS (SI)(AX*4), Y13

	VMOVUPS     (R12)(AX*4), Y8
	VFMADD231PS Y12, Y8, Y0
	VFMADD231PS Y13, Y8, Y1
	VMOVUPS     (R13)(AX*4), Y9
	VFMADD231PS Y12, Y9, Y2
	VFMADD231PS Y13, Y9, Y3
	VMOVUPS     (R14)(AX*4), Y10
	VFMADD231PS Y12, Y10, Y4
	VFMADD231PS Y13, Y10, Y5
	VMOVUPS     (R15)(AX*4), Y11
	VFMADD231PS Y12, Y11, Y6
	VFMADD231PS Y13, Y11, Y7

	ADDQ $8, AX
	CMPQ AX, CX
	JLT  d8

	// Eight horizontal sums into the four rows, two scores each.
	VEXTRACTF128 $1, Y0, X8
	VADDPS       X8, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VEXTRACTF128 $1, Y1, X9
	VADDPS       X9, X1, X1
	VHADDPS      X1, X1, X1
	VHADDPS      X1, X1, X1
	MOVQ         R9, R11
	VMOVSS       X0, (R11)(DX*4)
	VMOVSS       X1, 4(R11)(DX*4)

	VEXTRACTF128 $1, Y2, X8
	VADDPS       X8, X2, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	VEXTRACTF128 $1, Y3, X9
	VADDPS       X9, X3, X3
	VHADDPS      X3, X3, X3
	VHADDPS      X3, X3, X3
	ADDQ         R10, R11
	VMOVSS       X2, (R11)(DX*4)
	VMOVSS       X3, 4(R11)(DX*4)

	VEXTRACTF128 $1, Y4, X8
	VADDPS       X8, X4, X4
	VHADDPS      X4, X4, X4
	VHADDPS      X4, X4, X4
	VEXTRACTF128 $1, Y5, X9
	VADDPS       X9, X5, X5
	VHADDPS      X5, X5, X5
	VHADDPS      X5, X5, X5
	ADDQ         R10, R11
	VMOVSS       X4, (R11)(DX*4)
	VMOVSS       X5, 4(R11)(DX*4)

	VEXTRACTF128 $1, Y6, X8
	VADDPS       X8, X6, X6
	VHADDPS      X6, X6, X6
	VHADDPS      X6, X6, X6
	VEXTRACTF128 $1, Y7, X9
	VADDPS       X9, X7, X7
	VHADDPS      X7, X7, X7
	VHADDPS      X7, X7, X7
	ADDQ         R10, R11
	VMOVSS       X6, (R11)(DX*4)
	VMOVSS       X7, 4(R11)(DX*4)

	LEAQ (SI)(R8*1), DI    // on to the next pair
	ADDQ $2, DX

k2cond:
	CMPQ DX, BX
	JLT  k2

	// A last key on its own, when there is an odd number of them.
	MOVQ n+24(FP), R11
	CMPQ DX, R11
	JGE  done

	VXORPS Y0, Y0, Y0
	VXORPS Y2, Y2, Y2
	VXORPS Y4, Y4, Y4
	VXORPS Y6, Y6, Y6
	XORQ   AX, AX

e8:
	VMOVUPS     (DI)(AX*4), Y12
	VMOVUPS     (R12)(AX*4), Y8
	VFMADD231PS Y12, Y8, Y0
	VMOVUPS     (R13)(AX*4), Y9
	VFMADD231PS Y12, Y9, Y2
	VMOVUPS     (R14)(AX*4), Y10
	VFMADD231PS Y12, Y10, Y4
	VMOVUPS     (R15)(AX*4), Y11
	VFMADD231PS Y12, Y11, Y6
	ADDQ        $8, AX
	CMPQ        AX, CX
	JLT         e8

	MOVQ         R9, R11
	VEXTRACTF128 $1, Y0, X8
	VADDPS       X8, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VMOVSS       X0, (R11)(DX*4)
	ADDQ         R10, R11
	VEXTRACTF128 $1, Y2, X8
	VADDPS       X8, X2, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	VMOVSS       X2, (R11)(DX*4)
	ADDQ         R10, R11
	VEXTRACTF128 $1, Y4, X8
	VADDPS       X8, X4, X4
	VHADDPS      X4, X4, X4
	VHADDPS      X4, X4, X4
	VMOVSS       X4, (R11)(DX*4)
	ADDQ         R10, R11
	VEXTRACTF128 $1, Y6, X8
	VADDPS       X8, X6, X6
	VHADDPS      X6, X6, X6
	VHADDPS      X6, X6, X6
	VMOVSS       X6, (R11)(DX*4)

done:
	VZEROUPPER
	RET

// func mix4AVX2(dst *float32, dstStride int, v *float32, w *float32, wStride int, hd, n int)
//
// Four weighted sums of the same n value rows, one per query.
//
// The head is taken sixteen floats at a time so that four sums fit in eight
// registers with room for the values and their weights: each pass reads a
// quarter of the values and serves four queries with it, where a sum at a time
// would read all of them four times over.
//
// hd must be a multiple of sixteen, which sixty-four is.
TEXT ·mix4AVX2(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), R9
	MOVQ dstStride+8(FP), R10
	MOVQ v+16(FP), DI
	MOVQ w+24(FP), R11
	MOVQ wStride+32(FP), R12
	MOVQ hd+40(FP), CX
	MOVQ n+48(FP), R13

	SHLQ $2, R10           // between two destinations, in bytes
	SHLQ $2, R12           // between two rows of weights, in bytes
	MOVQ CX, R8
	SHLQ $2, R8            // one value row, in bytes

	XORQ BX, BX            // the sixteen floats of the head being summed

quarter:
	// The four running sums, two registers each.
	MOVQ   R9, SI
	VMOVUPS (SI)(BX*4), Y0
	VMOVUPS 32(SI)(BX*4), Y1
	ADDQ   R10, SI
	VMOVUPS (SI)(BX*4), Y2
	VMOVUPS 32(SI)(BX*4), Y3
	ADDQ   R10, SI
	VMOVUPS (SI)(BX*4), Y4
	VMOVUPS 32(SI)(BX*4), Y5
	ADDQ   R10, SI
	VMOVUPS (SI)(BX*4), Y6
	VMOVUPS 32(SI)(BX*4), Y7

	MOVQ DI, SI            // the value rows
	MOVQ R11, R14          // the four weights of this key
	XORQ DX, DX

row:
	VMOVUPS (SI)(BX*4), Y12
	VMOVUPS 32(SI)(BX*4), Y13

	MOVQ         R14, R15
	VBROADCASTSS (R15), Y8
	VFMADD231PS  Y12, Y8, Y0
	VFMADD231PS  Y13, Y8, Y1
	ADDQ         R12, R15
	VBROADCASTSS (R15), Y9
	VFMADD231PS  Y12, Y9, Y2
	VFMADD231PS  Y13, Y9, Y3
	ADDQ         R12, R15
	VBROADCASTSS (R15), Y10
	VFMADD231PS  Y12, Y10, Y4
	VFMADD231PS  Y13, Y10, Y5
	ADDQ         R12, R15
	VBROADCASTSS (R15), Y11
	VFMADD231PS  Y12, Y11, Y6
	VFMADD231PS  Y13, Y11, Y7

	ADDQ $4, R14           // the next key's four weights
	ADDQ R8, SI
	INCQ DX
	CMPQ DX, R13
	JLT  row

	MOVQ    R9, SI
	VMOVUPS Y0, (SI)(BX*4)
	VMOVUPS Y1, 32(SI)(BX*4)
	ADDQ    R10, SI
	VMOVUPS Y2, (SI)(BX*4)
	VMOVUPS Y3, 32(SI)(BX*4)
	ADDQ    R10, SI
	VMOVUPS Y4, (SI)(BX*4)
	VMOVUPS Y5, 32(SI)(BX*4)
	ADDQ    R10, SI
	VMOVUPS Y6, (SI)(BX*4)
	VMOVUPS Y7, 32(SI)(BX*4)

	ADDQ $16, BX
	CMPQ BX, CX
	JLT  quarter

	VZEROUPPER
	RET
