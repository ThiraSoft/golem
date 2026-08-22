//go:build amd64

#include "textflag.h"

// func dotBF16x4AVX2(row *uint16, x *float32, stride, n int, out *float32)
//
// One row of bfloat16 weights against four float32 vectors at once, the
// vectors laid stride floats apart.
//
// Four columns rather than one, because the one-column kernel is not limited
// by its arithmetic. Each eight weights there cost a widening, a shift, a load
// of the activation and one multiply-accumulate: three instructions to feed
// one. Widening the row once and feeding four columns with it makes that ten
// instructions for four, which is what it takes to keep both fused-multiply
// units busy on this machine.
//
// Eight accumulators — four columns, two halves of sixteen elements — because
// the multiply-accumulate has a latency of four cycles and a throughput of
// two: fewer chains than that and the units wait on themselves.
TEXT ·dotBF16x4AVX2(SB), NOSPLIT, $0-40
	MOVQ row+0(FP), SI
	MOVQ x+8(FP), DI
	MOVQ stride+16(FP), R8
	MOVQ n+24(FP), CX
	MOVQ out+32(FP), R9

	SHLQ $2, R8            // the stride in bytes

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	// The four columns, addressed from four bases so that one index walks all
	// of them.
	MOVQ DI, R12
	LEAQ (DI)(R8*1), R13
	LEAQ (DI)(R8*2), R14
	LEAQ (R13)(R8*2), R15

	XORQ AX, AX
	MOVQ CX, DX
	ANDQ $-16, DX
	JMP  b16cond

b16:
	PREFETCHT0 2048(SI)(AX*2)
	VPMOVZXWD (SI)(AX*2), Y8
	VPSLLD    $16, Y8, Y8
	VPMOVZXWD 16(SI)(AX*2), Y9
	VPSLLD    $16, Y9, Y9

	VMOVUPS     (R12)(AX*4), Y10
	VFMADD231PS Y10, Y8, Y0
	VMOVUPS     32(R12)(AX*4), Y11
	VFMADD231PS Y11, Y9, Y1

	VMOVUPS     (R13)(AX*4), Y12
	VFMADD231PS Y12, Y8, Y2
	VMOVUPS     32(R13)(AX*4), Y13
	VFMADD231PS Y13, Y9, Y3

	VMOVUPS     (R14)(AX*4), Y10
	VFMADD231PS Y10, Y8, Y4
	VMOVUPS     32(R14)(AX*4), Y11
	VFMADD231PS Y11, Y9, Y5

	VMOVUPS     (R15)(AX*4), Y12
	VFMADD231PS Y12, Y8, Y6
	VMOVUPS     32(R15)(AX*4), Y13
	VFMADD231PS Y13, Y9, Y7

	ADDQ $16, AX

b16cond:
	CMPQ AX, DX
	JLT  b16

	// Eight at a time for what is left of a row that is not a multiple of
	// sixteen, then one at a time.
	MOVQ CX, DX
	ANDQ $-8, DX
	JMP  b8cond

b8:
	VPMOVZXWD (SI)(AX*2), Y8
	VPSLLD    $16, Y8, Y8
	VMOVUPS     (R12)(AX*4), Y10
	VFMADD231PS Y10, Y8, Y0
	VMOVUPS     (R13)(AX*4), Y11
	VFMADD231PS Y11, Y8, Y2
	VMOVUPS     (R14)(AX*4), Y12
	VFMADD231PS Y12, Y8, Y4
	VMOVUPS     (R15)(AX*4), Y13
	VFMADD231PS Y13, Y8, Y6
	ADDQ $8, AX

b8cond:
	CMPQ AX, DX
	JLT  b8

	JMP b1cond

b1:
	MOVWLZX (SI)(AX*2), BX
	SHLL    $16, BX
	VMOVD   BX, X8
	VBROADCASTSS X8, X8

	VMOVSS (R12)(AX*4), X10
	VFMADD231SS X10, X8, X1
	VMOVSS (R13)(AX*4), X11
	VFMADD231SS X11, X8, X3
	VMOVSS (R14)(AX*4), X12
	VFMADD231SS X12, X8, X5
	VMOVSS (R15)(AX*4), X13
	VFMADD231SS X13, X8, X7
	INCQ    AX

b1cond:
	CMPQ AX, CX
	JLT  b1

	// Four horizontal reductions, one per column.
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y5, Y4, Y4
	VADDPS Y7, Y6, Y6

	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VMOVSS       X0, (R9)

	VEXTRACTF128 $1, Y2, X3
	VADDPS       X3, X2, X2
	VHADDPS      X2, X2, X2
	VHADDPS      X2, X2, X2
	VMOVSS       X2, 4(R9)

	VEXTRACTF128 $1, Y4, X5
	VADDPS       X5, X4, X4
	VHADDPS      X4, X4, X4
	VHADDPS      X4, X4, X4
	VMOVSS       X4, 8(R9)

	VEXTRACTF128 $1, Y6, X7
	VADDPS       X7, X6, X6
	VHADDPS      X6, X6, X6
	VHADDPS      X6, X6, X6
	VMOVSS       X6, 12(R9)

	VZEROUPPER
	RET
