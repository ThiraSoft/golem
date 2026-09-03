//go:build amd64

#include "textflag.h"

// One chunk of 32 weights: the pairs of products land in int16, the group's
// scale multiplies them, and the int32 result joins the superblock's total.
// Bytes 0..15 of the chunk share one scale and bytes 16..31 the next, which is
// exactly how VPMADDUBSW splits its lanes — Q6_K's macro, at two bits.
#define CHUNK2(QV, AOFF, SLO, SHI) \
	VPBROADCASTW SLO(R13), Y12  \
	VPBROADCASTW SHI(R13), Y13  \
	VPBLENDD     $0xF0, Y13, Y12, Y12 \
	VPMADDUBSW   AOFF(DI), QV, Y11 \
	VPMADDWD     Y12, Y11, Y11  \
	VPADDD       Y11, Y14, Y14

// One chunk's magnitudes: two bits a weight out of a window of thirty-two
// bytes, at the chunk's own bit position. The word shift drags a neighbour's
// bits into the top of each byte and the mask drops them, which is the same
// argument Q3_K's kernel makes.
#define MAGS2(WOFF, SH, AOFF, SLO, SHI) \
	VMOVDQU WOFF(SI), Y7 \
	VPSRLW  $SH, Y7, Y7  \
	VPAND   Y0, Y7, Y7   \
	CHUNK2(Y7, AOFF, SLO, SHI)

// And the chunk that sits at bit nought, where there is no shift to do.
#define MAGS2_0(WOFF, AOFF, SLO, SHI) \
	VMOVDQU WOFF(SI), Y7 \
	VPAND   Y0, Y7, Y7   \
	CHUNK2(Y7, AOFF, SLO, SHI)

// func dotQ2_KAVX2(w *byte, q *int8, bsums *int16, scales *float32, n int) float32
//
// One Q2_K row against one Q8_K activation, in integers.
//
// A superblock is 84 bytes: sixteen bytes each packing a four-bit scale and a
// four-bit minimum, sixty-four bytes of two-bit quants, and the two fp16 those
// nibbles are measured in.
//
// The minimum is what makes this kernel shorter than Q3_K's rather than longer.
// A Q2_K weight is scale*q less a minimum that does not depend on the weight,
// and the minimum is constant over a group of sixteen — which is exactly the
// span a Q8_K activation carries a sum for. So the whole second term is one
// VPMADDWD of the sixteen minima against the sixteen sums the caller brought,
// once a superblock, and no part of it touches a weight. Q3_K and Q6_K pay
// their recentring the same way; the difference is that theirs is a constant
// and this one is sixteen different numbers.
TEXT ·dotQ2_KAVX2(SB), NOSPLIT, $32-44
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ bsums+16(FP), R9
	MOVQ scales+24(FP), R10
	MOVQ n+32(FP), CX
	MOVQ SP, R13            // the sixteen scales, widened, live here

	SHRQ $8, CX             // CX = number of superblocks
	XORQ BX, BX

	MOVL         $0x03030303, R8
	VMOVD        R8, X0
	VPBROADCASTD X0, Y0     // the two-bit mask, for a magnitude
	MOVL         $0x0F0F0F0F, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1     // the nibble mask, for a scale and a minimum

	VXORPS X15, X15, X15    // the running total
	JMP    cond

superblock:
	PREFETCHT0 256(SI)
	VXORPS     Y14, Y14, Y14

	// The sixteen bytes at the front of the block are a scale low and a
	// minimum high. Both come out unsigned and neither is biased, which is
	// what separates this format from every other K-quant here.
	VMOVDQU   (SI), X2
	VPAND     X1, X2, X3
	VPSRLW    $4, X2, X4
	VPAND     X1, X4, X4
	VPMOVZXBW X3, Y3
	VMOVDQU   Y3, (R13)

	// The minima against the activation's own group sums, once for the whole
	// superblock: sixteen products, eight int32 lanes.
	VPMOVZXBW X4, Y4
	VMOVDQU   (R9), Y5
	VPMADDWD  Y5, Y4, Y4

	MAGS2_0(16, 0, 0, 2)
	MAGS2(16, 2, 32, 4, 6)
	MAGS2(16, 4, 64, 8, 10)
	MAGS2(16, 6, 96, 12, 14)
	MAGS2_0(48, 128, 16, 18)
	MAGS2(48, 2, 160, 20, 22)
	MAGS2(48, 4, 192, 24, 26)
	MAGS2(48, 6, 224, 28, 30)

	VEXTRACTI128 $1, Y14, X11
	VPADDD       X11, X14, X14
	VPSHUFD      $0x4E, X14, X11
	VPADDD       X11, X14, X14
	VPSHUFD      $0xB1, X14, X11
	VPADDD       X11, X14, X14

	VEXTRACTI128 $1, Y4, X11
	VPADDD       X11, X4, X4
	VPSHUFD      $0x4E, X4, X11
	VPADDD       X11, X4, X4
	VPSHUFD      $0xB1, X4, X11
	VPADDD       X11, X4, X4

	VCVTDQ2PS X14, X14
	VCVTDQ2PS X4, X4
	MOVWLZX   80(SI), R8    // d, and dmin two bytes after it. Both loaded
	VMOVD     R8, X12       // narrow: the block is 84 bytes and a wider read
	VCVTPH2PS X12, X12      // would run past the last one in the mapping
	MOVWLZX   82(SI), R8
	VMOVD     R8, X13
	VCVTPH2PS X13, X13
	VMULSS    X12, X14, X14
	VMULSS    X13, X4, X4
	VSUBSS    X4, X14, X14
	VMULSS    (R10)(BX*4), X14, X14
	VADDSS    X14, X15, X15

	ADDQ $84, SI
	ADDQ $256, DI
	ADDQ $32, R9
	INCQ BX

cond:
	CMPQ BX, CX
	JLT  superblock

	VMOVSS X15, ret+40(FP)
	VZEROUPPER
	RET
