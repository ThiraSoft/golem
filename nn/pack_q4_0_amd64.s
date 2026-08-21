//go:build amd64

#include "textflag.h"

// The packed Q4_0 product: eight rows of weights at once, one lane each.
//
// The group holds a block as eight fp16 scales and four chunks of thirty-two
// bytes. A chunk covers eight inputs; its low nibbles are four rows and its
// high nibbles are four more, so one load and three instructions give the
// eight rows' weights for those inputs.
//
// The activation's eight bytes are broadcast to the four slots of a chunk, and
// VPMADDUBSW multiplies them there. Its int16 pairs are summed over the four
// chunks before anything widens them: a lane holds at most four times two
// products of a nibble by an activation byte, which is fifteen thousand and
// change, so int16 carries it and the widening costs one instruction a block
// instead of four. VPMADDWD then leaves two int32 per row, and one VPHADDD
// folds the two accumulators into eight sums in ascending row order — which is
// what nn/pack_q4_0.go lays the rows out for.
//
// What is left is one conversion, one fused multiply for the scales and one
// for the correction, per block and per column, for all eight rows.

DATA packMask<>+0x00(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA packMask<>+0x08(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA packMask<>+0x10(SB)/8, $0x0F0F0F0F0F0F0F0F
DATA packMask<>+0x18(SB)/8, $0x0F0F0F0F0F0F0F0F
GLOBL packMask<>(SB), RODATA|NOPTR, $32

DATA packOnes<>+0x00(SB)/8, $0x0001000100010001
DATA packOnes<>+0x08(SB)/8, $0x0001000100010001
DATA packOnes<>+0x10(SB)/8, $0x0001000100010001
DATA packOnes<>+0x18(SB)/8, $0x0001000100010001
GLOBL packOnes<>(SB), RODATA|NOPTR, $32

// The four chunks of one block against one column, into eight row sums.
#define CHUNKS(QOFF) \
	VPBROADCASTQ QOFF(DI), Y14      \
	VPMADDUBSW   Y14, Y0, Y12       \
	VPMADDUBSW   Y14, Y1, Y13       \
	VPBROADCASTQ (QOFF+8)(DI), Y14  \
	VPMADDUBSW   Y14, Y2, Y15       \
	VPADDW       Y15, Y12, Y12      \
	VPMADDUBSW   Y14, Y3, Y15       \
	VPADDW       Y15, Y13, Y13      \
	VPBROADCASTQ (QOFF+16)(DI), Y14 \
	VPMADDUBSW   Y14, Y4, Y15       \
	VPADDW       Y15, Y12, Y12      \
	VPMADDUBSW   Y14, Y5, Y15       \
	VPADDW       Y15, Y13, Y13      \
	VPBROADCASTQ (QOFF+24)(DI), Y14 \
	VPMADDUBSW   Y14, Y6, Y15       \
	VPADDW       Y15, Y12, Y12      \
	VPMADDUBSW   Y14, Y7, Y15       \
	VPADDW       Y15, Y13, Y13      \
	VPMADDWD     packOnes<>(SB), Y12, Y12 \
	VPMADDWD     packOnes<>(SB), Y13, Y13 \
	VPHADDD      Y13, Y12, Y12      \
	VCVTDQ2PS    Y12, Y12

// One column of a block: its sums, then the two scales that turn them into
// products — the row's and the activation's, and the recentring the activation
// carries.
#define COLUMNP(QOFF, SOFF, ACC) \
	CHUNKS(QOFF)                    \
	VBROADCASTSS SOFF(DX), Y14      \
	VMULPS       (R12), Y14, Y14    \
	VFMADD231PS  Y14, Y12, ACC      \
	VBROADCASTSS SOFF(R9), Y14      \
	VFNMADD231PS (R12), Y14, ACC

// The eight rows' weights of one block, unpacked into Y0..Y7: chunk g in
// Y2g and Y2g+1, low nibbles then high.
#define UNPACKP \
	VCVTPH2PS (SI), Y15               \
	VMOVUPS   Y15, (R12)              \
	VMOVDQU   16(SI), Y15             \
	VPAND     packMask<>(SB), Y15, Y0 \
	VPSRLW    $4, Y15, Y15            \
	VPAND     packMask<>(SB), Y15, Y1 \
	VMOVDQU   48(SI), Y15             \
	VPAND     packMask<>(SB), Y15, Y2 \
	VPSRLW    $4, Y15, Y15            \
	VPAND     packMask<>(SB), Y15, Y3 \
	VMOVDQU   80(SI), Y15             \
	VPAND     packMask<>(SB), Y15, Y4 \
	VPSRLW    $4, Y15, Y15            \
	VPAND     packMask<>(SB), Y15, Y5 \
	VMOVDQU   112(SI), Y15            \
	VPAND     packMask<>(SB), Y15, Y6 \
	VPSRLW    $4, Y15, Y15            \
	VPAND     packMask<>(SB), Y15, Y7

// func dotPackedQ4_0x4AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)
//
// Four columns against one group of eight rows. The state is four sets of
// eight lanes, a lane to a row, and it is a running total rather than
// something to be folded: the horizontal add happens once per block, inside.
TEXT ·dotPackedQ4_0x4AVX2(SB), NOSPLIT, $32-64
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ scales+16(FP), DX
	MOVQ corr+24(FP), R9
	MOVQ n+32(FP), CX
	MOVQ stride+40(FP), R10
	MOVQ state+48(FP), R11
	MOVQ SP, R12            // the block's eight row scales live here

	SHRQ $5, CX             // blocks
	MOVQ R10, R13
	SHLQ $5, R13            // bytes from one block of activations to the next
	SHLQ $2, R10            // bytes from one block of scales to the next

	MOVQ  mode+56(FP), AX
	TESTQ $1, AX
	JNZ   zeroP4

	VMOVUPS (R11), Y8
	VMOVUPS 32(R11), Y9
	VMOVUPS 64(R11), Y10
	VMOVUPS 96(R11), Y11
	JMP     condP4

zeroP4:
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11

condP4:
	CMPQ CX, $0
	JLE  doneP4

blockP4:
	UNPACKP
	COLUMNP(0, 0, Y8)
	COLUMNP(32, 4, Y9)
	COLUMNP(64, 8, Y10)
	COLUMNP(96, 12, Y11)

	ADDQ $144, SI
	ADDQ R13, DI
	ADDQ R10, DX
	ADDQ R10, R9
	DECQ CX
	JGT  blockP4

doneP4:
	VMOVUPS Y8, (R11)
	VMOVUPS Y9, 32(R11)
	VMOVUPS Y10, 64(R11)
	VMOVUPS Y11, 96(R11)
	VZEROUPPER
	RET

// func dotPackedQ4_0AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)
//
// The same for one column, instruction for instruction, so that a batch of one
// and a batch of sixty-four give the same float.
TEXT ·dotPackedQ4_0AVX2(SB), NOSPLIT, $32-64
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ scales+16(FP), DX
	MOVQ corr+24(FP), R9
	MOVQ n+32(FP), CX
	MOVQ stride+40(FP), R10
	MOVQ state+48(FP), R11
	MOVQ SP, R12

	SHRQ $5, CX
	MOVQ R10, R13
	SHLQ $5, R13
	SHLQ $2, R10

	MOVQ  mode+56(FP), AX
	TESTQ $1, AX
	JNZ   zeroP1

	VMOVUPS (R11), Y8
	JMP     condP1

zeroP1:
	VXORPS Y8, Y8, Y8

condP1:
	CMPQ CX, $0
	JLE  doneP1

blockP1:
	UNPACKP
	COLUMNP(0, 0, Y8)

	ADDQ $144, SI
	ADDQ R13, DI
	ADDQ R10, DX
	ADDQ R10, R9
	DECQ CX
	JGT  blockP1

doneP1:
	VMOVUPS Y8, (R11)
	VZEROUPPER
	RET
