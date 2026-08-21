//go:build amd64

#include "textflag.h"

// One chunk of 32 weights, against two activations.
//
// The unpacking is what this saves. The six-bit magnitudes come out of the row
// once, in Y7, and both columns multiply the same register: the row is read
// from memory once, shifted and masked once, and only the products are done
// twice. The group's scale vector is built once too, since it belongs to the
// weights and not to the activation.
#define CHUNK2(QV, AOFF, SLO, SHI) \
	VPBROADCASTW SLO(R13), Y12  \
	VPBROADCASTW SHI(R13), Y13  \
	VPBLENDD     $0xF0, Y13, Y12, Y12 \
	VPMADDUBSW   AOFF(DI), QV, Y11 \
	VPMADDWD     Y12, Y11, Y11  \
	VPADDD       Y11, Y14, Y14  \
	VPMADDUBSW   AOFF(R14), QV, Y13 \
	VPMADDWD     Y12, Y13, Y13  \
	VPADDD       Y13, Y2, Y2

// One half of a superblock: 128 weights, drawn from 64 bytes of low nibbles and
// 32 bytes of high bit pairs, in the four groups ggml writes them in.
#define HALF2(L0, L1, H, A0, A1, A2, A3, S0, S1, S2, S3, S4, S5, S6, S7) \
	VMOVDQU L0(SI), Y4       \
	VMOVDQU L1(SI), Y5       \
	VMOVDQU H(SI), Y6        \
	VPAND   Y0, Y4, Y7       \
	VPAND   Y1, Y6, Y8       \
	VPSLLW  $4, Y8, Y8       \
	VPOR    Y8, Y7, Y7       \
	CHUNK2(Y7, A0, S0, S1)   \
	VPAND   Y0, Y5, Y7       \
	VPSRLW  $2, Y6, Y9       \
	VPAND   Y1, Y9, Y8       \
	VPSLLW  $4, Y8, Y8       \
	VPOR    Y8, Y7, Y7       \
	CHUNK2(Y7, A1, S2, S3)   \
	VPSRLW  $4, Y4, Y7       \
	VPAND   Y0, Y7, Y7       \
	VPSRLW  $4, Y6, Y9       \
	VPAND   Y1, Y9, Y8       \
	VPSLLW  $4, Y8, Y8       \
	VPOR    Y8, Y7, Y7       \
	CHUNK2(Y7, A2, S4, S5)   \
	VPSRLW  $4, Y5, Y7       \
	VPAND   Y0, Y7, Y7       \
	VPSRLW  $6, Y6, Y9       \
	VPAND   Y1, Y9, Y8       \
	VPSLLW  $4, Y8, Y8       \
	VPOR    Y8, Y7, Y7       \
	CHUNK2(Y7, A3, S6, S7)

// func dotQ6_Kx2AVX2(w *byte, q0, q1 *int8, bsums0, bsums1 *int16, scales0, scales1 *float32, n int, out *float32)
//
// One Q6_K row against two Q8_K activations, in integers, writing two floats.
//
// Everything about the arithmetic is dotQ6_KAVX2's: the same products in the
// same order, the same recentring by 32 taken from the sums the activation
// carries, the same two floating multiplies per superblock. What differs is
// that a row pays for its unpacking once instead of twice, which is what the
// output head of a model wants when several conversations are drawing at once.
TEXT ·dotQ6_Kx2AVX2(SB), NOSPLIT, $32-72
	MOVQ w+0(FP), SI
	MOVQ q0+8(FP), DI
	MOVQ q1+16(FP), R14
	MOVQ bsums0+24(FP), R9
	MOVQ bsums1+32(FP), R15
	MOVQ scales0+40(FP), R10
	MOVQ scales1+48(FP), R11
	MOVQ n+56(FP), CX
	MOVQ SP, R13            // the sixteen scales, widened, live here

	SHRQ $8, CX             // CX = number of superblocks
	XORQ BX, BX

	MOVL         $0x0F0F0F0F, R8
	VMOVD        R8, X0
	VPBROADCASTD X0, Y0     // the low-nibble mask
	MOVL         $0x03030303, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1     // the high-bit pair mask

	VXORPS X15, X15, X15    // the running total of the first column
	VXORPS X10, X10, X10    // and of the second
	JMP    cond

superblock:
	PREFETCHT0 512(SI)
	PREFETCHT0 576(SI)
	VXORPS     Y14, Y14, Y14
	VXORPS     Y2, Y2, Y2
	VPMOVSXBW  192(SI), Y3
	VMOVDQU    Y3, (R13)

	HALF2(0, 32, 128, 0, 32, 64, 96, 0, 2, 4, 6, 8, 10, 12, 14)
	HALF2(64, 96, 160, 128, 160, 192, 224, 16, 18, 20, 22, 24, 26, 28, 30)

	// The recentring, one column at a time: 32 x sum over the groups of
	// scale x sum(q), where the sums belong to the activation.
	VMOVDQU  (R9), Y9
	VPMADDWD Y9, Y3, Y9
	VPSLLD   $5, Y9, Y9
	VPSUBD   Y9, Y14, Y14
	VMOVDQU  (R15), Y8
	VPMADDWD Y8, Y3, Y8
	VPSLLD   $5, Y8, Y8
	VPSUBD   Y8, Y2, Y2

	// The superblock's own fp16 scale, loaded narrow: the block is 210 bytes
	// and a wider read would run past the end of the last one in the mapping.
	MOVWLZX   208(SI), R8
	VMOVD     R8, X12
	VCVTPH2PS X12, X12

	VEXTRACTI128 $1, Y14, X11
	VPADDD       X11, X14, X14
	VPSHUFD      $0x4E, X14, X11
	VPADDD       X11, X14, X14
	VPSHUFD      $0xB1, X14, X11
	VPADDD       X11, X14, X14
	VCVTDQ2PS    X14, X14
	VMULSS       X12, X14, X14
	VMULSS       (R10)(BX*4), X14, X14
	VADDSS       X14, X15, X15

	VEXTRACTI128 $1, Y2, X11
	VPADDD       X11, X2, X2
	VPSHUFD      $0x4E, X2, X11
	VPADDD       X11, X2, X2
	VPSHUFD      $0xB1, X2, X11
	VPADDD       X11, X2, X2
	VCVTDQ2PS    X2, X2
	VMULSS       X12, X2, X2
	VMULSS       (R11)(BX*4), X2, X2
	VADDSS       X2, X10, X10

	ADDQ $210, SI
	ADDQ $256, DI
	ADDQ $256, R14
	ADDQ $32, R9
	ADDQ $32, R15
	INCQ BX

cond:
	CMPQ BX, CX
	JLT  superblock

	MOVQ   out+64(FP), AX
	VMOVSS X15, (AX)
	VMOVSS X10, 4(AX)
	VZEROUPPER
	RET
