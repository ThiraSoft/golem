//go:build amd64

#include "textflag.h"

// The Q8_0 product against a Q8_0 activation, in integers.
//
// A block is 34 bytes: an fp16 scale, then thirty-two signed weights, one to a
// byte. Against dot_q4_0's kernel two things are gone and one is new. Gone is
// the unpacking — the bytes are the weights — and gone is the correction, the
// term Q4_0 carries because its nibbles are unsigned numbers waiting to have
// eight taken off them, where a Q8_0 weight is already signed.
//
// New is the sign. VPMADDUBSW multiplies an unsigned first operand by a signed
// second, and both operands here are signed, so the row is split: its
// magnitudes go in unsigned, and its signs are moved onto the activation
// first. That is two VPSIGNB a block — one for |w|, one for y*sign(w) —
// against the sixteen bytes of masking and shifting Q4_0 spends on nibbles.
// The identity is |w| * (y * sign(w)) = w * y, exactly, in integers, with
// nothing rounded.
//
// Like the Q4_0 kernels, nothing is reduced horizontally inside the loop: the
// eight partial sums of a block are converted to floats, scaled by the product
// of the two scales, and added to a running vector. One reduction happens at
// the end of a row, and the eight lanes survive between calls so that a row
// cut into stretches gives what the whole row gives.
//
// The activations of a batch are interleaved block by block and, within a
// block, column by column, so `stride` — the number of columns — separates one
// block from the next. A batch of one is the layout of a lone vector, and the
// same instructions walk it.

// One accumulator down to one float. Q4_0's REDUCE ends by subtracting a
// correction; there is none here, so this is that sequence without its last
// subtraction, and the additions are in the same order — which is what lets
// fold() in Go and this agree on where the rounding falls.
#define REDUCE0(ACC, LOW, SLOT) \
	VEXTRACTF128 $1, ACC, X7  \
	VADDPS       X7, LOW, LOW \
	VMOVHLPS     LOW, LOW, X7 \
	VADDPS       X7, LOW, LOW \
	VSHUFPS      $0x01, LOW, LOW, X7 \
	VADDSS       X7, LOW, LOW \
	VMOVSS       LOW, SLOT(R11)

// func dotQ8_0AVX2(w *byte, q *int8, scales *float32, n, stride int, state *float32, mode int)
//
// One column against one row. There is no four- or eight-column form: what
// those buy for Q4_0 is unpacking a row once for several products, and a Q8_0
// row is not unpacked at all. The two VPSIGNB per block would be shared, and
// they are two instructions against the thirty-two bytes of weights the loop
// must read either way.
TEXT ·dotQ8_0AVX2(SB), NOSPLIT, $0-56
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ scales+16(FP), DX
	MOVQ n+24(FP), CX
	MOVQ stride+32(FP), R10
	MOVQ state+40(FP), R11

	SHRQ $5, CX   // blocks, not inputs
	MOVQ R10, R13
	SHLQ $5, R13  // stride * 32: one block of the interleaved activation
	SHLQ $2, R10  // stride * 4: one block of its scales

	MOVL         $0x00010001, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1  // the pairs of int16 that VPMADDWD adds

	MOVQ  mode+48(FP), AX
	TESTQ $1, AX
	JNZ   zero

	VMOVUPS (R11), Y10  // the running total, eight lanes wide
	JMP     cond

zero:
	VXORPS Y10, Y10, Y10
	JMP    cond

block:
	PREFETCHT0 1024(SI)

	VMOVDQU 2(SI), Y2    // the row's thirty-two weights, signed
	VPSIGNB Y2, Y2, Y3   // |w|, unsigned as VPMADDUBSW wants
	VMOVDQU (DI), Y13    // the activation, signed
	VPSIGNB Y2, Y13, Y13 // y * sign(w)

	VPMADDUBSW Y13, Y3, Y13 // |w| * (y * sign(w)), pairs into int16
	VPMADDWD   Y1, Y13, Y13 // pairs into int32
	VCVTDQ2PS  Y13, Y13

	VCVTPH2PS    (SI), X7   // the block's scale
	VMULSS       (DX), X7, X8
	VBROADCASTSS X8, Y8
	VFMADD231PS  Y8, Y13, Y10

	ADDQ $34, SI
	ADDQ R13, DI
	ADDQ R10, DX
	DECQ CX

cond:
	CMPQ CX, $0
	JGT  block

	TESTQ $2, AX
	JNZ   reduce

	VMOVUPS Y10, (R11)
	VZEROUPPER
	RET

reduce:
	REDUCE0(Y10, X10, 0)
	VZEROUPPER
	RET
