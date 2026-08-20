//go:build amd64

#include "textflag.h"

// One chunk of 32 weights: the pairs of products land in int16, the group's
// scale multiplies them, and the int32 result joins the superblock's total.
// Bytes 0..15 of the chunk share one scale and bytes 16..31 the next, which is
// exactly how VPMADDUBSW splits its lanes.
#define CHUNK(QV, AOFF, SLO, SHI) \
	VPBROADCASTW SLO(R13), Y12  \
	VPBROADCASTW SHI(R13), Y13  \
	VPBLENDD     $0xF0, Y13, Y12, Y12 \
	VPMADDUBSW   AOFF(DI), QV, Y11 \
	VPMADDWD     Y12, Y11, Y11  \
	VPADDD       Y11, Y14, Y14

// One half of a superblock: 128 weights, drawn from 64 bytes of low nibbles and
// 32 bytes of high bit pairs, in the four groups ggml writes them in.
#define HALF(L0, L1, H, A0, A1, A2, A3, S0, S1, S2, S3, S4, S5, S6, S7) \
	VMOVDQU L0(SI), Y4       \
	VMOVDQU L1(SI), Y5       \
	VMOVDQU H(SI), Y6        \
	VPAND   Y0, Y4, Y7       \
	VPAND   Y1, Y6, Y8       \
	VPSLLW  $4, Y8, Y8       \
	VPOR    Y8, Y7, Y7       \
	CHUNK(Y7, A0, S0, S1)    \
	VPAND   Y0, Y5, Y7       \
	VPSRLW  $2, Y6, Y9       \
	VPAND   Y1, Y9, Y8       \
	VPSLLW  $4, Y8, Y8       \
	VPOR    Y8, Y7, Y7       \
	CHUNK(Y7, A1, S2, S3)    \
	VPSRLW  $4, Y4, Y7       \
	VPAND   Y0, Y7, Y7       \
	VPSRLW  $4, Y6, Y9       \
	VPAND   Y1, Y9, Y8       \
	VPSLLW  $4, Y8, Y8       \
	VPOR    Y8, Y7, Y7       \
	CHUNK(Y7, A2, S4, S5)    \
	VPSRLW  $4, Y5, Y7       \
	VPAND   Y0, Y7, Y7       \
	VPSRLW  $6, Y6, Y9       \
	VPAND   Y1, Y9, Y8       \
	VPSLLW  $4, Y8, Y8       \
	VPOR    Y8, Y7, Y7       \
	CHUNK(Y7, A3, S6, S7)

// func dotQ6_KAVX2(w *byte, q *int8, bsums *int16, scales *float32, n int) float32
//
// One Q6_K row against one Q8_K activation, in integers.
//
// A superblock is 210 bytes: 128 bytes of low nibbles, 64 bytes carrying two
// high bits per weight, 16 signed scales covering 16 weights each, and one fp16
// scale for the whole superblock.
//
// The six-bit magnitudes stay unsigned in 0..63, which is what VPMADDUBSW wants
// of its first operand, and the recentring by 32 never touches a weight: it is
// 32 x scale x sum(q) per group of sixteen, taken at the end of the superblock
// from the sums the activation carries. Only the two scales are floating point,
// and they are applied once per superblock.
TEXT ·dotQ6_KAVX2(SB), NOSPLIT, $32-44
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ bsums+16(FP), R9
	MOVQ scales+24(FP), R10
	MOVQ n+32(FP), CX
	MOVQ SP, R13            // the sixteen scales, widened, live here

	SHRQ $8, CX             // CX = number of superblocks
	XORQ BX, BX

	MOVL         $0x0F0F0F0F, R8
	VMOVD        R8, X0
	VPBROADCASTD X0, Y0     // the low-nibble mask
	MOVL         $0x03030303, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1     // the high-bit pair mask

	VXORPS X15, X15, X15    // the running total
	JMP    cond

superblock:
	PREFETCHT0 512(SI)
	PREFETCHT0 576(SI)
	VXORPS     Y14, Y14, Y14
	VPMOVSXBW 192(SI), Y3
	VMOVDQU   Y3, (R13)

	HALF(0, 32, 128, 0, 32, 64, 96, 0, 2, 4, 6, 8, 10, 12, 14)
	HALF(64, 96, 160, 128, 160, 192, 224, 16, 18, 20, 22, 24, 26, 28, 30)

	// The recentring: 32 x sum over the groups of scale x sum(q).
	VMOVDQU  (R9), Y10
	VPMADDWD Y10, Y3, Y10
	VPSLLD   $5, Y10, Y10
	VPSUBD   Y10, Y14, Y14

	VEXTRACTI128 $1, Y14, X11
	VPADDD       X11, X14, X14
	VPSHUFD      $0x4E, X14, X11
	VPADDD       X11, X14, X14
	VPSHUFD      $0xB1, X14, X11
	VPADDD       X11, X14, X14

	VCVTDQ2PS X14, X14
	MOVWLZX   208(SI), R8   // the superblock's fp16 scale, loaded narrow: the
	VMOVD     R8, X12       // block is 210 bytes and a wider read would run
	VCVTPH2PS X12, X12      // past the end of the last one in the mapping
	VMULSS    X12, X14, X14
	VMULSS    (R10)(BX*4), X14, X14
	VADDSS    X14, X15, X15

	ADDQ $210, SI
	ADDQ $256, DI
	ADDQ $32, R9
	INCQ BX

cond:
	CMPQ BX, CX
	JLT  superblock

	VMOVSS X15, ret+40(FP)
	VZEROUPPER
	RET
