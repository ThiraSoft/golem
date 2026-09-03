//go:build amd64

#include "textflag.h"

// The four shifts the sixteen six-bit scales are packed at, one to a dword
// lane: scales[k]'s two high bits come from byte k%4 of sc[8:12] at 2*(k/4).
DATA q3kShifts<>+0(SB)/4, $0
DATA q3kShifts<>+4(SB)/4, $2
DATA q3kShifts<>+8(SB)/4, $4
DATA q3kShifts<>+12(SB)/4, $6
GLOBL q3kShifts<>(SB), RODATA|NOPTR, $16

// One chunk of 32 weights: the pairs of products land in int16, the group's
// scale multiplies them, and the int32 result joins the superblock's total.
// Bytes 0..15 of the chunk share one scale and bytes 16..31 the next, which is
// exactly how VPMADDUBSW splits its lanes — the same arrangement Q6_K's kernel
// has, and for the same reason: a group is sixteen weights and a lane is
// sixteen bytes.
#define CHUNK3(QV, AOFF, SLO, SHI) \
	VPBROADCASTW SLO(R13), Y12  \
	VPBROADCASTW SHI(R13), Y13  \
	VPBLENDD     $0xF0, Y13, Y12, Y12 \
	VPMADDUBSW   AOFF(DI), QV, Y11 \
	VPMADDWD     Y12, Y11, Y11  \
	VPADDD       Y11, Y14, Y14

// One chunk's magnitudes, drawn from two bits of a qs window and one bit of the
// superblock's hmask.
//
// QSRC is the window — the first thirty-two bytes of qs carry chunks nought to
// three and the second thirty-two carry four to seven — SH is the bit position
// the chunk sits at inside it, and HSH is which bit of hmask is the chunk's.
//
// Both shifts are word shifts over byte lanes, which drags a neighbour's bits
// into the top of each byte. The masks that follow are what makes that
// harmless: after a shift of SH the wanted pair is at bits 0 and 1 and the
// intruders are at 6 and 7, and 0x03 keeps only the pair. The third bit is the
// same argument at one bit and 0x01.
#define MAGS(QSRC, SH, HSH) \
	VPSRLW $SH, QSRC, Y7 \
	VPAND  Y1, Y7, Y7    \
	VPSRLW $HSH, Y6, Y8  \
	VPAND  Y2, Y8, Y8    \
	VPSLLW $2, Y8, Y8    \
	VPOR   Y8, Y7, Y7

// func dotQ3_KAVX2(w *byte, q *int8, bsums *int16, scales *float32, n int) float32
//
// One Q3_K row against one Q8_K activation, in integers.
//
// A superblock is 110 bytes: 32 bytes of third bits, 64 bytes carrying two bits
// per weight, 12 bytes holding sixteen six-bit signed scales, and one fp16
// scale for the whole superblock.
//
// The three-bit magnitudes stay unsigned in 0..7, which is what VPMADDUBSW
// wants of its first operand, and the recentring by four never touches a
// weight: it is 4 x scale x sum(q) per group of sixteen, taken at the end of
// the superblock from the sums the activation carries. That is Q6_K's kernel
// with a different unpacking and a different shift, which is what the format
// actually differs by.
//
// The twelve packed scale bytes are undone in six instructions rather than the
// sixteen turns of a scalar loop, which at 256 weights a superblock is not a
// detail: the loop was as much work as the products it fed.
TEXT ·dotQ3_KAVX2(SB), NOSPLIT, $32-44
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
	VPBROADCASTD X0, Y0     // the low-nibble mask, for the scales
	MOVL         $0x03030303, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1     // the two-bit mask, for a magnitude's low pair
	MOVL         $0x01010101, R8
	VMOVD        R8, X2
	VPBROADCASTD X2, Y2     // the one-bit mask, for a magnitude's third bit
	MOVL         $0x20202020, R8
	VMOVD        R8, X3
	VPBROADCASTD X3, Y10    // thirty-two, which every scale is stored above

	VXORPS X15, X15, X15    // the running total
	JMP    cond

superblock:
	PREFETCHT0 256(SI)
	PREFETCHT0 320(SI)
	VXORPS     Y14, Y14, Y14

	// The scales. sc[0:8] holds every scale's low four bits — the first eight
	// in the low nibbles and the last eight in the high ones — and sc[8:12]
	// holds all sixteen pairs of high bits, four to a byte at four shifts. Both
	// are loaded narrow: a superblock is 110 bytes and a sixteen-byte read from
	// offset 96 would run two past the last one in the mapping.
	MOVQ        96(SI), R8
	VMOVQ       R8, X4
	VPSRLW      $4, X4, X5
	VPUNPCKLQDQ X5, X4, X4
	VPAND       X0, X4, X4      // the low four bits, in the order of the output

	MOVL         104(SI), R8
	VMOVD        R8, X5
	VPBROADCASTD X5, X5         // the same four bytes for each group of four
	VPSRLVD      q3kShifts<>(SB), X5, X5
	VPAND        X1, X5, X5
	VPSLLW       $4, X5, X5      // the two high bits, into bits four and five
	VPOR         X5, X4, X4
	VPSUBB       X10, X4, X4     // less thirty-two, which makes them signed
	VPMOVSXBW    X4, Y3
	VMOVDQU      Y3, (R13)

	VMOVDQU (SI), Y6        // the third bits, one per weight, all eight chunks
	VMOVDQU 32(SI), Y4      // the low pairs of chunks nought to three
	VMOVDQU 64(SI), Y5      // and of chunks four to seven

	MAGS(Y4, 0, 0)
	CHUNK3(Y7, 0, 0, 2)
	MAGS(Y4, 2, 1)
	CHUNK3(Y7, 32, 4, 6)
	MAGS(Y4, 4, 2)
	CHUNK3(Y7, 64, 8, 10)
	MAGS(Y4, 6, 3)
	CHUNK3(Y7, 96, 12, 14)
	MAGS(Y5, 0, 4)
	CHUNK3(Y7, 128, 16, 18)
	MAGS(Y5, 2, 5)
	CHUNK3(Y7, 160, 20, 22)
	MAGS(Y5, 4, 6)
	CHUNK3(Y7, 192, 24, 26)
	MAGS(Y5, 6, 7)
	CHUNK3(Y7, 224, 28, 30)

	// The recentring: 4 x sum over the groups of scale x sum(q).
	VMOVDQU  (R9), Y9
	VPMADDWD Y9, Y3, Y9
	VPSLLD   $2, Y9, Y9
	VPSUBD   Y9, Y14, Y14

	VEXTRACTI128 $1, Y14, X11
	VPADDD       X11, X14, X14
	VPSHUFD      $0x4E, X14, X11
	VPADDD       X11, X14, X14
	VPSHUFD      $0xB1, X14, X11
	VPADDD       X11, X14, X14

	VCVTDQ2PS X14, X14
	MOVWLZX   108(SI), R8   // the superblock's fp16 scale, loaded narrow: the
	VMOVD     R8, X12       // block is 110 bytes and a wider read would run
	VCVTPH2PS X12, X12      // past the end of the last one in the mapping
	VMULSS    X12, X14, X14
	VMULSS    (R10)(BX*4), X14, X14
	VADDSS    X14, X15, X15

	ADDQ $110, SI
	ADDQ $256, DI
	ADDQ $32, R9
	INCQ BX

cond:
	CMPQ BX, CX
	JLT  superblock

	VMOVSS X15, ret+40(FP)
	VZEROUPPER
	RET
