//go:build amd64

#include "textflag.h"

// The ternary products against a Q8_K activation. dot_ternary.go says what
// the activation looks like and why; dotPQ2_0Go and dotPTQ1_0Go are the same
// integers, summed in another order.

// The multipliers that bring a PTQ1_0 trit to the front of its byte, for the
// two stages that are eight bytes wide: those bytes are put in both halves of
// a register and each half takes its own digit. The last register is the
// eight-byte stage's fifth digit beside the two high bytes' four.
DATA ptqMul01<>+0(SB)/2, $1
DATA ptqMul01<>+2(SB)/2, $1
DATA ptqMul01<>+4(SB)/2, $1
DATA ptqMul01<>+6(SB)/2, $1
DATA ptqMul01<>+8(SB)/2, $1
DATA ptqMul01<>+10(SB)/2, $1
DATA ptqMul01<>+12(SB)/2, $1
DATA ptqMul01<>+14(SB)/2, $1
DATA ptqMul01<>+16(SB)/2, $3
DATA ptqMul01<>+18(SB)/2, $3
DATA ptqMul01<>+20(SB)/2, $3
DATA ptqMul01<>+22(SB)/2, $3
DATA ptqMul01<>+24(SB)/2, $3
DATA ptqMul01<>+26(SB)/2, $3
DATA ptqMul01<>+28(SB)/2, $3
DATA ptqMul01<>+30(SB)/2, $3
GLOBL ptqMul01<>(SB), RODATA, $32

DATA ptqMul23<>+0(SB)/2, $9
DATA ptqMul23<>+2(SB)/2, $9
DATA ptqMul23<>+4(SB)/2, $9
DATA ptqMul23<>+6(SB)/2, $9
DATA ptqMul23<>+8(SB)/2, $9
DATA ptqMul23<>+10(SB)/2, $9
DATA ptqMul23<>+12(SB)/2, $9
DATA ptqMul23<>+14(SB)/2, $9
DATA ptqMul23<>+16(SB)/2, $27
DATA ptqMul23<>+18(SB)/2, $27
DATA ptqMul23<>+20(SB)/2, $27
DATA ptqMul23<>+22(SB)/2, $27
DATA ptqMul23<>+24(SB)/2, $27
DATA ptqMul23<>+26(SB)/2, $27
DATA ptqMul23<>+28(SB)/2, $27
DATA ptqMul23<>+30(SB)/2, $27
GLOBL ptqMul23<>(SB), RODATA, $32

DATA ptqMul4h<>+0(SB)/2, $81
DATA ptqMul4h<>+2(SB)/2, $81
DATA ptqMul4h<>+4(SB)/2, $81
DATA ptqMul4h<>+6(SB)/2, $81
DATA ptqMul4h<>+8(SB)/2, $81
DATA ptqMul4h<>+10(SB)/2, $81
DATA ptqMul4h<>+12(SB)/2, $81
DATA ptqMul4h<>+14(SB)/2, $81
DATA ptqMul4h<>+16(SB)/2, $1
DATA ptqMul4h<>+18(SB)/2, $1
DATA ptqMul4h<>+20(SB)/2, $3
DATA ptqMul4h<>+22(SB)/2, $3
DATA ptqMul4h<>+24(SB)/2, $9
DATA ptqMul4h<>+26(SB)/2, $9
DATA ptqMul4h<>+28(SB)/2, $27
DATA ptqMul4h<>+30(SB)/2, $27
GLOBL ptqMul4h<>(SB), RODATA, $32

// The horizontal sum of Y15 into X15.
#define HSUM15 \
	VEXTRACTF128 $1, Y15, X14 \
	VADDPS       X14, X15, X15 \
	VPSHUFD      $0x4E, X15, X14 \
	VADDPS       X14, X15, X15 \
	VPSHUFD      $0xB1, X15, X14 \
	VADDPS       X14, X15, X15

// A block's int32 lanes in Y4, less the activation's sum over the block, put
// into floats and added into Y15 at the block's scale times the superblock's.
// The fp16 is loaded on its own: a wider read would run past the last block of
// the mapping.
#define FINISH(DOFF) \
	VMOVD        (R9)(BX*4), X5 \
	VPSUBD       Y5, Y4, Y4 \
	VCVTDQ2PS    Y4, Y4 \
	MOVWLZX      DOFF(SI), R8 \
	VMOVD        R8, X6 \
	VCVTPH2PS    X6, X6 \
	MOVQ         BX, R11 \
	SHRQ         $1, R11 \
	VMULSS       (R10)(R11*4), X6, X6 \
	VBROADCASTSS X6, Y6 \
	VFMADD231PS  Y6, Y4, Y15

// func dotPQ2_0AVX2(w *byte, q *int8, sums *int32, scales *float32, blocks int) float32
//
// A block is an fp16 and thirty-two bytes of codes. The four planes of codes
// are the bytes shifted by 0, 2, 4 and 6 and masked, and the activation is
// already in plane order, so each plane is one VPMADDUBSW against the next
// thirty-two bytes of it. Four planes of at most 3 x 127 x 2 stay inside int16.
TEXT ·dotPQ2_0AVX2(SB), NOSPLIT, $0-44
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ sums+16(FP), R9
	MOVQ scales+24(FP), R10
	MOVQ blocks+32(FP), CX
	XORQ BX, BX

	MOVL         $0x03030303, R8
	VMOVD        R8, X0
	VPBROADCASTD X0, Y0     // the two-bit mask
	MOVL         $0x00010001, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1     // ones, to widen int16 pairs to int32

	VXORPS Y15, Y15, Y15
	JMP    pq2cond

pq2block:
	// Ahead of the hardware's own prefetch, and around the caches: a row is
	// read once a token. Worth eight percent of a token on the processor,
	// where these products are the memory's speed and nothing else.
	PREFETCHNTA 1024(SI)
	VMOVDQU    2(SI), Y2
	VPAND      Y0, Y2, Y3
	VPMADDUBSW (DI), Y3, Y4
	VPSRLW     $2, Y2, Y3
	VPAND      Y0, Y3, Y3
	VPMADDUBSW 32(DI), Y3, Y5
	VPADDW     Y5, Y4, Y4
	VPSRLW     $4, Y2, Y3
	VPAND      Y0, Y3, Y3
	VPMADDUBSW 64(DI), Y3, Y5
	VPADDW     Y5, Y4, Y4
	VPSRLW     $6, Y2, Y3
	VPAND      Y0, Y3, Y3
	VPMADDUBSW 96(DI), Y3, Y5
	VPADDW     Y5, Y4, Y4
	VPMADDWD   Y1, Y4, Y4
	FINISH(0)

	ADDQ $34, SI
	ADDQ $128, DI
	INCQ BX

pq2cond:
	CMPQ BX, CX
	JLT  pq2block

	HSUM15
	VMOVSS X15, ret+40(FP)
	VZEROUPPER
	RET

// One digit of sixteen PTQ1_0 bytes held as words in SRC: times 3^t, kept to
// a byte, times three, the top byte. Then against sixteen activations,
// widened, into the block's lanes.
#define TRIT(SRC, MUL, AOFF) \
	VPMULLW   MUL, SRC, Y9 \
	VPAND     Y0, Y9, Y9   \
	VPMULLW   Y1, Y9, Y9   \
	VPSRLW    $8, Y9, Y9   \
	VPMOVSXBW AOFF(DI), Y7 \
	VPMADDWD  Y7, Y9, Y9   \
	VPADDD    Y9, Y4, Y4

// func dotPTQ1_0AVX2(w *byte, q *int8, sums *int32, scales *float32, blocks int) float32
//
// A block is twenty-four bytes of five trits each, two of four, and an fp16.
// The first sixteen bytes give weights 16t+m, the next eight 80+8t+m, the last
// two 120+2t+h: every digit of a stage is a run of the activation, so it
// stays in natural order and only the bytes are moved.
TEXT ·dotPTQ1_0AVX2(SB), NOSPLIT, $0-44
	MOVQ w+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ sums+16(FP), R9
	MOVQ scales+24(FP), R10
	MOVQ blocks+32(FP), CX
	XORQ BX, BX

	MOVL         $0x00FF00FF, R8
	VMOVD        R8, X0
	VPBROADCASTD X0, Y0     // a byte, in each word
	MOVL         $0x00030003, R8
	VMOVD        R8, X1
	VPBROADCASTD X1, Y1
	MOVL         $0x00090009, R8
	VMOVD        R8, X2
	VPBROADCASTD X2, Y2
	MOVL         $0x001B001B, R8
	VMOVD        R8, X3
	VPBROADCASTD X3, Y3
	MOVL         $0x00510051, R8
	VMOVD        R8, X13
	VPBROADCASTD X13, Y13

	VXORPS Y15, Y15, Y15
	JMP    ptqcond

ptqblock:
	PREFETCHNTA 1024(SI)
	// The first digit needs no multiplier to bring it forward.
	VPMOVZXBW (SI), Y8
	VPMULLW   Y1, Y8, Y9
	VPSRLW    $8, Y9, Y9
	VPMOVSXBW (DI), Y7
	VPMADDWD  Y7, Y9, Y4
	TRIT(Y8, Y1, 16)
	TRIT(Y8, Y2, 32)
	TRIT(Y8, Y3, 48)
	TRIT(Y8, Y13, 64)

	VPMOVZXBW   16(SI), X10
	VINSERTI128 $1, X10, Y10, Y11
	TRIT(Y11, ptqMul01<>(SB), 80)
	TRIT(Y11, ptqMul23<>(SB), 96)
	MOVWLZX      24(SI), R8
	VMOVD        R8, X12
	VPMOVZXBW    X12, X12
	VPBROADCASTD X12, X12
	VINSERTI128  $1, X12, Y10, Y11
	TRIT(Y11, ptqMul4h<>(SB), 112)
	FINISH(26)

	ADDQ $28, SI
	ADDQ $128, DI
	INCQ BX

ptqcond:
	CMPQ BX, CX
	JLT  ptqblock

	HSUM15
	VMOVSS X15, ret+40(FP)
	VZEROUPPER
	RET
