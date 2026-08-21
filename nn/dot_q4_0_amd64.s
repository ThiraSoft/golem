//go:build amd64

#include "textflag.h"

// The Q4_0 product against a Q8_0 activation, in integers.
//
// A block is 18 bytes: an fp16 scale, then 32 weights packed two per byte, the
// low nibble holding weight j and the high nibble weight j+16. The nibbles stay
// unsigned in 0..15 here, because VPMADDUBSW insists on an unsigned first
// operand; the recentring by eight is not undone in the loop at all. It is
// 8 x scale x sum(q) per block, which depends on the activation alone and was
// computed once when the activation was quantized.
//
// Nothing is reduced horizontally inside the loops either: the eight partial
// sums of a block are converted to floats, scaled, and added to a running
// vector. One reduction happens at the end of the row.
//
// The activations of a batch are interleaved, block by block and within a block
// column by column, so `stride` — the number of columns in the batch — is what
// separates one block from the next. A batch of one leaves the layout of a lone
// vector, and the same instructions walk it.

// One column of a block: the pairs of products land in int16, an int32 sum, a
// float, and the block's two scales.
#define COLUMN(QOFF, SOFF, ACC) \
	VPMADDUBSW   QOFF(DI), Y2, Y13 \
	VPMADDWD     Y1, Y13, Y13      \
	VCVTDQ2PS    Y13, Y13          \
	VBROADCASTSS SOFF(SP), Y14     \
	VFMADD231PS  Y14, Y13, ACC

// One accumulator down to one float, less its correction lane.
#define REDUCE(ACC, LOW, CORR, SLOT) \
	VEXTRACTF128 $1, ACC, X7  \
	VADDPS       X7, LOW, LOW \
	VMOVHLPS     LOW, LOW, X7 \
	VADDPS       X7, LOW, LOW \
	VSHUFPS      $0x01, LOW, LOW, X7 \
	VADDSS       X7, LOW, LOW \
	VSUBSS       CORR, LOW, LOW \
	VMOVSS       LOW, SLOT(R11)

// The nibbles of one block, unsigned, low sixteen then high sixteen.
#define UNPACK \
	VMOVDQU     2(SI), X8   \
	VPAND       X0, X8, X9  \
	VPSRLW      $4, X8, X8  \
	VPAND       X0, X8, X8  \
	VINSERTI128 $1, X8, Y9, Y2

// func dotQ4_0x4AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)
//
// Four columns against one row. The row is unpacked once and the block's fp16
// scale converted once for the four: at one column per pass, two thirds of the
// work in this loop is spent on the weights rather than on the product. The
// eight-column kernel below carries that further; this one finishes a batch
// that is not a multiple of eight.
//
// The eight lanes of each column, and the four corrections, live in the state
// the caller passes: a row cut into several stretches of input accumulates into
// the same lanes, in the same order, as a row done in one go. That is what lets
// the loops be tiled for the cache without changing an answer.
TEXT ·dotQ4_0x4AVX2(SB), NOSPLIT, $16-64
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ scales+16(FP), DX
	MOVQ corr+24(FP), R9
	MOVQ n+32(FP), CX
	MOVQ stride+40(FP), R10
	MOVQ state+48(FP), R11
	MOVQ SP, R12            // the four scale products live here

	SHRQ $5, CX             // CX = number of blocks
	MOVQ R10, R13
	SHLQ $5, R13            // bytes from one block of activations to the next
	SHLQ $2, R10            // bytes from one block of scales to the next

	MOVL         $0x0F0F0F0F, R8
	VMOVD        R8, X0
	VPBROADCASTD X0, X0     // the low-nibble mask
	MOVL         $0x00010001, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1     // ones, as int16

	// mode says where the sums come from and where they go: bit 0 starts them
	// at zero rather than at what the state holds, bit 1 folds them into four
	// products rather than storing the lanes for the next stretch of input.
	MOVQ  mode+56(FP), AX
	TESTQ $1, AX
	JNZ   zero4

	VMOVUPS (R11), Y3
	VMOVUPS 32(R11), Y4
	VMOVUPS 64(R11), Y5
	VMOVUPS 96(R11), Y6
	VMOVUPS 128(R11), X12   // the four running corrections
	JMP     cond4

zero4:
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS X12, X12, X12
	JMP    cond4

block4:
	UNPACK
	VCVTPH2PS    (SI), X7
	VBROADCASTSS X7, X10

	VMOVUPS (DX), X8
	VMULPS  X10, X8, X8
	VMOVUPS X8, (R12)       // scale x scale, one per column

	VMOVUPS (R9), X11
	VMULPS  X10, X11, X11
	VADDPS  X11, X12, X12

	COLUMN(0, 0, Y3)
	COLUMN(32, 4, Y4)
	COLUMN(64, 8, Y5)
	COLUMN(96, 12, Y6)

	ADDQ $18, SI
	ADDQ R13, DI
	ADDQ R10, DX
	ADDQ R10, R9
	DECQ CX

cond4:
	CMPQ CX, $0
	JGT  block4

	TESTQ $2, AX
	JNZ   fold4

	VMOVUPS Y3, (R11)
	VMOVUPS Y4, 32(R11)
	VMOVUPS Y5, 64(R11)
	VMOVUPS Y6, 96(R11)
	VMOVUPS X12, 128(R11)
	VZEROUPPER
	RET

	// Asked to finish, the kernel folds the lanes itself and writes the four
	// products where the lanes were. It is the same arithmetic in the same
	// order as the fold in Go, so a row done in one call and a row done in
	// several give the same float.
fold4:
	VMOVUPS X12, (R12)
	VMOVSS  (R12), X13
	REDUCE(Y3, X3, X13, 0)
	VMOVSS  4(R12), X13
	REDUCE(Y4, X4, X13, 4)
	VMOVSS  8(R12), X13
	REDUCE(Y5, X5, X13, 8)
	VMOVSS  12(R12), X13
	REDUCE(Y6, X6, X13, 12)

	VZEROUPPER
	RET

// func dotQ4_0AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)
//
// One column against one row, for the columns a batch has left over and for
// generation, where there is only ever one. Its state is eight lanes and one
// correction, and the caller folds them.
TEXT ·dotQ4_0AVX2(SB), NOSPLIT, $0-64
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ scales+16(FP), DX
	MOVQ corr+24(FP), R9
	MOVQ n+32(FP), CX
	MOVQ stride+40(FP), R10
	MOVQ state+48(FP), R11

	SHRQ $5, CX
	MOVQ R10, R13
	SHLQ $5, R13
	SHLQ $2, R10

	MOVL         $0x0F0F0F0F, R8
	VMOVD        R8, X0
	VPBROADCASTD X0, X0
	MOVL         $0x00010001, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1

	MOVQ  mode+56(FP), AX
	TESTQ $1, AX
	JNZ   zero1

	VMOVUPS (R11), Y10      // the running total, eight lanes wide
	VMOVSS  32(R11), X12    // the running correction
	JMP     cond

zero1:
	VXORPS Y10, Y10, Y10
	VXORPS X12, X12, X12
	JMP    cond

block:
	PREFETCHT0 1024(SI)
	UNPACK
	VCVTPH2PS    (SI), X7
	VMULSS       (DX), X7, X8
	VBROADCASTSS X8, Y8

	VPMADDUBSW (DI), Y2, Y13
	VPMADDWD   Y1, Y13, Y13
	VCVTDQ2PS  Y13, Y13
	VFMADD231PS Y8, Y13, Y10

	VMULSS (R9), X7, X9
	VADDSS X9, X12, X12

	ADDQ $18, SI
	ADDQ R13, DI
	ADDQ R10, DX
	ADDQ R10, R9
	DECQ CX

cond:
	CMPQ CX, $0
	JGT  block

	TESTQ $2, AX
	JNZ   fold1

	VMOVUPS Y10, (R11)
	VMOVSS  X12, 32(R11)
	VZEROUPPER
	RET

fold1:
	REDUCE(Y10, X10, X12, 0)
	VZEROUPPER
	RET

// One column of a block, into an accumulator whose lanes belong to that
// column alone: the same six instructions as COLUMN, reading its scale
// product from the scratch the block wrote.
#define COLUMN8(QOFF, SOFF, ACC) \
	VPMADDUBSW   QOFF(DI), Y2, Y13 \
	VPMADDWD     Y1, Y13, Y13      \
	VCVTDQ2PS    Y13, Y13          \
	VBROADCASTSS SOFF(R12), Y14    \
	VFMADD231PS  Y14, Y13, ACC

// One accumulator down to one float, with X14 as the scratch REDUCE keeps in
// X7 — X7 is an accumulator here.
#define REDUCE8(ACC, LOW, CORR, SLOT) \
	VEXTRACTF128 $1, ACC, X14  \
	VADDPS       X14, LOW, LOW \
	VMOVHLPS     LOW, LOW, X14 \
	VADDPS       X14, LOW, LOW \
	VSHUFPS      $0x01, LOW, LOW, X14 \
	VADDSS       X14, LOW, LOW \
	VSUBSS       CORR, LOW, LOW \
	VMOVSS       LOW, SLOT(R11)

// The nibbles of one block, in the scratch registers the eight-column kernel
// has left: X8 and X9 are accumulators there.
#define UNPACK8 \
	VMOVDQU     2(SI), X13   \
	VPAND       X0, X13, X14 \
	VPSRLW      $4, X13, X13 \
	VPAND       X0, X13, X13 \
	VINSERTI128 $1, X13, Y14, Y2

// func dotQ4_0x8AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)
//
// Eight columns against one row. What the four-column kernel does for the
// unpacking of a row, this does twice over: the nibbles, the fp16 scale and
// the correction are read and converted once for eight products instead of
// four, and at that width they are no longer what the loop spends its time on.
//
// The state is the eight columns' eight lanes, then the eight corrections.
TEXT ·dotQ4_0x8AVX2(SB), NOSPLIT, $32-64
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ scales+16(FP), DX
	MOVQ corr+24(FP), R9
	MOVQ n+32(FP), CX
	MOVQ stride+40(FP), R10
	MOVQ state+48(FP), R11
	MOVQ SP, R12            // the eight scale products live here

	SHRQ $5, CX
	MOVQ R10, R13
	SHLQ $5, R13
	SHLQ $2, R10

	MOVL         $0x0F0F0F0F, R8
	VMOVD        R8, X0
	VPBROADCASTD X0, X0
	MOVL         $0x00010001, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1

	MOVQ  mode+56(FP), AX
	TESTQ $1, AX
	JNZ   zero8

	VMOVUPS (R11), Y3
	VMOVUPS 32(R11), Y4
	VMOVUPS 64(R11), Y5
	VMOVUPS 96(R11), Y6
	VMOVUPS 128(R11), Y7
	VMOVUPS 160(R11), Y8
	VMOVUPS 192(R11), Y9
	VMOVUPS 224(R11), Y10
	VMOVUPS 256(R11), Y12  // the eight running corrections
	JMP     cond8

zero8:
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y12, Y12, Y12
	JMP    cond8

block8:
	UNPACK8
	VCVTPH2PS    (SI), X11
	VBROADCASTSS X11, Y11

	VMOVUPS (DX), Y13
	VMULPS  Y11, Y13, Y13
	VMOVUPS Y13, (R12)

	// The correction stays a multiply and an add: the one-column kernel adds
	// it that way, and the two have to agree to the last bit.
	VMOVUPS (R9), Y13
	VMULPS  Y11, Y13, Y13
	VADDPS  Y13, Y12, Y12

	COLUMN8(0, 0, Y3)
	COLUMN8(32, 4, Y4)
	COLUMN8(64, 8, Y5)
	COLUMN8(96, 12, Y6)
	COLUMN8(128, 16, Y7)
	COLUMN8(160, 20, Y8)
	COLUMN8(192, 24, Y9)
	COLUMN8(224, 28, Y10)

	ADDQ $18, SI
	ADDQ R13, DI
	ADDQ R10, DX
	ADDQ R10, R9
	DECQ CX

cond8:
	CMPQ CX, $0
	JGT  block8

	TESTQ $2, AX
	JNZ   fold8

	VMOVUPS Y3, (R11)
	VMOVUPS Y4, 32(R11)
	VMOVUPS Y5, 64(R11)
	VMOVUPS Y6, 96(R11)
	VMOVUPS Y7, 128(R11)
	VMOVUPS Y8, 160(R11)
	VMOVUPS Y9, 192(R11)
	VMOVUPS Y10, 224(R11)
	VMOVUPS Y12, 256(R11)
	VZEROUPPER
	RET

fold8:
	VMOVUPS Y12, (R12)
	VMOVSS  (R12), X13
	REDUCE8(Y3, X3, X13, 0)
	VMOVSS  4(R12), X13
	REDUCE8(Y4, X4, X13, 4)
	VMOVSS  8(R12), X13
	REDUCE8(Y5, X5, X13, 8)
	VMOVSS  12(R12), X13
	REDUCE8(Y6, X6, X13, 12)
	VMOVSS  16(R12), X13
	REDUCE8(Y7, X7, X13, 16)
	VMOVSS  20(R12), X13
	REDUCE8(Y8, X8, X13, 20)
	VMOVSS  24(R12), X13
	REDUCE8(Y9, X9, X13, 24)
	VMOVSS  28(R12), X13
	REDUCE8(Y10, X10, X13, 28)

	VZEROUPPER
	RET
