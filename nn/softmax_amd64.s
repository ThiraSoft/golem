//go:build amd64

#include "textflag.h"

// The constants of ggml_v_expf, each replicated eight times so that AVX2 can
// take them as a memory operand: this instruction set has no broadcast folded
// into the operand the way the next one does.
DATA expc<>+0x00(SB)/4, $0x3fb8aa3b  // log2(e)
DATA expc<>+0x04(SB)/4, $0x3fb8aa3b
DATA expc<>+0x08(SB)/4, $0x3fb8aa3b
DATA expc<>+0x0c(SB)/4, $0x3fb8aa3b
DATA expc<>+0x10(SB)/4, $0x3fb8aa3b
DATA expc<>+0x14(SB)/4, $0x3fb8aa3b
DATA expc<>+0x18(SB)/4, $0x3fb8aa3b
DATA expc<>+0x1c(SB)/4, $0x3fb8aa3b

DATA expc<>+0x20(SB)/4, $0x4b400000  // 1.5 * 2^23
DATA expc<>+0x24(SB)/4, $0x4b400000
DATA expc<>+0x28(SB)/4, $0x4b400000
DATA expc<>+0x2c(SB)/4, $0x4b400000
DATA expc<>+0x30(SB)/4, $0x4b400000
DATA expc<>+0x34(SB)/4, $0x4b400000
DATA expc<>+0x38(SB)/4, $0x4b400000
DATA expc<>+0x3c(SB)/4, $0x4b400000

DATA expc<>+0x40(SB)/4, $0x3f317200  // ln2, high part
DATA expc<>+0x44(SB)/4, $0x3f317200
DATA expc<>+0x48(SB)/4, $0x3f317200
DATA expc<>+0x4c(SB)/4, $0x3f317200
DATA expc<>+0x50(SB)/4, $0x3f317200
DATA expc<>+0x54(SB)/4, $0x3f317200
DATA expc<>+0x58(SB)/4, $0x3f317200
DATA expc<>+0x5c(SB)/4, $0x3f317200

DATA expc<>+0x60(SB)/4, $0x35bfbe8e  // ln2, low part
DATA expc<>+0x64(SB)/4, $0x35bfbe8e
DATA expc<>+0x68(SB)/4, $0x35bfbe8e
DATA expc<>+0x6c(SB)/4, $0x35bfbe8e
DATA expc<>+0x70(SB)/4, $0x35bfbe8e
DATA expc<>+0x74(SB)/4, $0x35bfbe8e
DATA expc<>+0x78(SB)/4, $0x35bfbe8e
DATA expc<>+0x7c(SB)/4, $0x35bfbe8e

DATA expc<>+0x80(SB)/4, $0x3f7ffff6  // c0
DATA expc<>+0x84(SB)/4, $0x3f7ffff6
DATA expc<>+0x88(SB)/4, $0x3f7ffff6
DATA expc<>+0x8c(SB)/4, $0x3f7ffff6
DATA expc<>+0x90(SB)/4, $0x3f7ffff6
DATA expc<>+0x94(SB)/4, $0x3f7ffff6
DATA expc<>+0x98(SB)/4, $0x3f7ffff6
DATA expc<>+0x9c(SB)/4, $0x3f7ffff6

DATA expc<>+0xa0(SB)/4, $0x3efffedb  // c1
DATA expc<>+0xa4(SB)/4, $0x3efffedb
DATA expc<>+0xa8(SB)/4, $0x3efffedb
DATA expc<>+0xac(SB)/4, $0x3efffedb
DATA expc<>+0xb0(SB)/4, $0x3efffedb
DATA expc<>+0xb4(SB)/4, $0x3efffedb
DATA expc<>+0xb8(SB)/4, $0x3efffedb
DATA expc<>+0xbc(SB)/4, $0x3efffedb

DATA expc<>+0xc0(SB)/4, $0x3e2aaf33  // c2
DATA expc<>+0xc4(SB)/4, $0x3e2aaf33
DATA expc<>+0xc8(SB)/4, $0x3e2aaf33
DATA expc<>+0xcc(SB)/4, $0x3e2aaf33
DATA expc<>+0xd0(SB)/4, $0x3e2aaf33
DATA expc<>+0xd4(SB)/4, $0x3e2aaf33
DATA expc<>+0xd8(SB)/4, $0x3e2aaf33
DATA expc<>+0xdc(SB)/4, $0x3e2aaf33

DATA expc<>+0xe0(SB)/4, $0x3d2b9f17  // c3
DATA expc<>+0xe4(SB)/4, $0x3d2b9f17
DATA expc<>+0xe8(SB)/4, $0x3d2b9f17
DATA expc<>+0xec(SB)/4, $0x3d2b9f17
DATA expc<>+0xf0(SB)/4, $0x3d2b9f17
DATA expc<>+0xf4(SB)/4, $0x3d2b9f17
DATA expc<>+0xf8(SB)/4, $0x3d2b9f17
DATA expc<>+0xfc(SB)/4, $0x3d2b9f17

DATA expc<>+0x100(SB)/4, $0x3c072010 // c4
DATA expc<>+0x104(SB)/4, $0x3c072010
DATA expc<>+0x108(SB)/4, $0x3c072010
DATA expc<>+0x10c(SB)/4, $0x3c072010
DATA expc<>+0x110(SB)/4, $0x3c072010
DATA expc<>+0x114(SB)/4, $0x3c072010
DATA expc<>+0x118(SB)/4, $0x3c072010
DATA expc<>+0x11c(SB)/4, $0x3c072010

DATA expc<>+0x120(SB)/4, $0x3f800000 // 1.0, whose bits are the exponent bias
DATA expc<>+0x124(SB)/4, $0x3f800000
DATA expc<>+0x128(SB)/4, $0x3f800000
DATA expc<>+0x12c(SB)/4, $0x3f800000
DATA expc<>+0x130(SB)/4, $0x3f800000
DATA expc<>+0x134(SB)/4, $0x3f800000
DATA expc<>+0x138(SB)/4, $0x3f800000
DATA expc<>+0x13c(SB)/4, $0x3f800000

DATA expc<>+0x140(SB)/4, $0xc2ae0000 // -87, below which the result is nothing
DATA expc<>+0x144(SB)/4, $0xc2ae0000
DATA expc<>+0x148(SB)/4, $0xc2ae0000
DATA expc<>+0x14c(SB)/4, $0xc2ae0000
DATA expc<>+0x150(SB)/4, $0xc2ae0000
DATA expc<>+0x154(SB)/4, $0xc2ae0000
DATA expc<>+0x158(SB)/4, $0xc2ae0000
DATA expc<>+0x15c(SB)/4, $0xc2ae0000

GLOBL expc<>(SB), RODATA|NOPTR, $352

// func softmaxAVX2(x *float32, n int)
//
// The maximum, the exponential of every difference from it, and the division
// by their sum — three sweeps, and the middle one is ggml_v_expf eight values
// at a time.
//
// This is the routine an attention row spends itself on. A vision tower scores
// a thousand keys for each of a thousand queries, twelve heads deep and
// sixteen blocks tall, so the exponential is called two hundred million times
// for one picture; a scalar library exponential, accurate to the last bit and
// branching to prove it, is the wrong instrument by an order of magnitude.
//
// An input more than eighty-seven below the maximum is held there, where the
// exponential has already reached what a float32 calls zero.
TEXT ·softmaxAVX2(SB), NOSPLIT, $0-16
	MOVQ x+0(FP), DI
	MOVQ n+8(FP), CX

	// The maximum, eight at a time.
	VBROADCASTSS expc<>+0x140(SB), Y0
	XORQ AX, AX
	MOVQ CX, DX
	ANDQ $-8, DX
	JMP  maxcond

max8:
	VMAXPS (DI)(AX*4), Y0, Y0
	ADDQ   $8, AX

maxcond:
	CMPQ AX, DX
	JLT  max8

	VEXTRACTF128 $1, Y0, X1
	VMAXPS       X1, X0, X0
	VMOVHLPS     X0, X0, X1
	VMAXPS       X1, X0, X0
	VSHUFPS      $1, X0, X0, X1
	VMAXPS       X1, X0, X0

max1:
	CMPQ   AX, CX
	JGE    maxdone
	VMOVSS (DI)(AX*4), X1
	VMAXSS X1, X0, X0
	INCQ   AX
	JMP    max1

maxdone:
	VBROADCASTSS X0, Y15         // the maximum, in every lane

	// The exponentials and their sum.
	VXORPS Y14, Y14, Y14         // the running sum
	XORQ   AX, AX
	JMP    expcond

exp8:
	VMOVUPS (DI)(AX*4), Y1
	VSUBPS  Y15, Y1, Y1
	VMAXPS  expc<>+0x140(SB), Y1, Y1   // held at -87

	VMOVAPS      Y1, Y2
	VMULPS       expc<>+0x00(SB), Y2, Y2
	VADDPS       expc<>+0x20(SB), Y2, Y2   // z = x*log2e + shift
	VSUBPS       expc<>+0x20(SB), Y2, Y3   // n = z - shift

	VMOVAPS      Y3, Y4
	VMULPS       expc<>+0x40(SB), Y4, Y4
	VSUBPS       Y4, Y1, Y4                // x - n*ln2hi
	VMOVAPS      Y3, Y5
	VMULPS       expc<>+0x60(SB), Y5, Y5
	VSUBPS       Y5, Y4, Y4                // b

	VPSLLD $23, Y2, Y5
	VPADDD expc<>+0x120(SB), Y5, Y5        // k = 2^n

	VMULPS Y4, Y4, Y6                      // u = b*b

	VMOVUPS     expc<>+0x100(SB), Y7
	VMULPS      Y4, Y7, Y7
	VADDPS      expc<>+0xe0(SB), Y7, Y7    // c4*b + c3
	VMOVUPS     expc<>+0xc0(SB), Y8
	VMULPS      Y4, Y8, Y8
	VADDPS      expc<>+0xa0(SB), Y8, Y8    // c2*b + c1
	VMULPS      Y6, Y7, Y7
	VADDPS      Y8, Y7, Y7                 // (c4*b+c3)*u + (c2*b+c1)
	VMULPS      Y6, Y7, Y7
	VMOVUPS     expc<>+0x80(SB), Y9
	VMULPS      Y4, Y9, Y9                 // c0*b
	VADDPS      Y9, Y7, Y7                 // j

	VMULPS  Y5, Y7, Y7
	VADDPS  Y5, Y7, Y7                     // j*k + k
	VMOVUPS Y7, (DI)(AX*4)
	VADDPS  Y7, Y14, Y14
	ADDQ    $8, AX

expcond:
	CMPQ AX, DX
	JLT  exp8

	// The horizontal sum, then whatever is left of the row one at a time.
	VEXTRACTF128 $1, Y14, X1
	VADDPS       X1, X14, X14
	VHADDPS      X14, X14, X14
	VHADDPS      X14, X14, X14

exp1:
	CMPQ   AX, CX
	JGE    expdone
	VMOVSS (DI)(AX*4), X1
	VSUBSS X15, X1, X1
	VMAXSS expc<>+0x140(SB), X1, X1

	VMOVAPS X1, X2
	VMULSS  expc<>+0x00(SB), X2, X2
	VADDSS  expc<>+0x20(SB), X2, X2
	VSUBSS  expc<>+0x20(SB), X2, X3

	VMOVAPS X3, X4
	VMULSS  expc<>+0x40(SB), X4, X4
	VSUBSS  X4, X1, X4
	VMOVAPS X3, X5
	VMULSS  expc<>+0x60(SB), X5, X5
	VSUBSS  X5, X4, X4

	VPSLLD $23, X2, X5
	VPADDD expc<>+0x120(SB), X5, X5

	VMULSS X4, X4, X6

	VMOVUPS expc<>+0x100(SB), X7
	VMULSS  X4, X7, X7
	VADDSS  expc<>+0xe0(SB), X7, X7
	VMOVUPS expc<>+0xc0(SB), X8
	VMULSS  X4, X8, X8
	VADDSS  expc<>+0xa0(SB), X8, X8
	VMULSS  X6, X7, X7
	VADDSS  X8, X7, X7
	VMULSS  X6, X7, X7
	VMOVUPS expc<>+0x80(SB), X9
	VMULSS  X4, X9, X9
	VADDSS  X9, X7, X7

	VMULSS X5, X7, X7
	VADDSS X5, X7, X7
	VMOVSS X7, (DI)(AX*4)
	VADDSS X7, X14, X14
	INCQ   AX
	JMP    exp1

expdone:
	// And the division, as a multiplication by the reciprocal.
	VMOVSS       expc<>+0x120(SB), X0
	VDIVSS       X14, X0, X0
	VBROADCASTSS X0, Y0

	XORQ AX, AX
	JMP  divcond

div8:
	VMOVUPS (DI)(AX*4), Y1
	VMULPS  Y0, Y1, Y1
	VMOVUPS Y1, (DI)(AX*4)
	ADDQ    $8, AX

divcond:
	CMPQ AX, DX
	JLT  div8

div1:
	CMPQ   AX, CX
	JGE    divdone
	VMOVSS (DI)(AX*4), X1
	VMULSS X0, X1, X1
	VMOVSS X1, (DI)(AX*4)
	INCQ   AX
	JMP    div1

divdone:
	VZEROUPPER
	RET

DATA gegluc<>+0x00(SB)/4, $0xbfd9db23  // -1.702, the quick GELU's slope
DATA gegluc<>+0x04(SB)/4, $0xbfd9db23
DATA gegluc<>+0x08(SB)/4, $0xbfd9db23
DATA gegluc<>+0x0c(SB)/4, $0xbfd9db23
DATA gegluc<>+0x10(SB)/4, $0xbfd9db23
DATA gegluc<>+0x14(SB)/4, $0xbfd9db23
DATA gegluc<>+0x18(SB)/4, $0xbfd9db23
DATA gegluc<>+0x1c(SB)/4, $0xbfd9db23

GLOBL gegluc<>(SB), RODATA|NOPTR, $32

// func gegluQuickAVX2(gate, up *float32, n int)
//
// gate = gate/(1 + exp(-1.702*gate)) * up, eight at a time, with the same
// exponential the softmax above uses — ggml's, which is what the feed forward
// of this tower is compared against.
//
// n must be a multiple of eight.
TEXT ·gegluQuickAVX2(SB), NOSPLIT, $0-24
	MOVQ gate+0(FP), DI
	MOVQ up+8(FP), SI
	MOVQ n+16(FP), CX
	XORQ AX, AX

g8:
	VMOVUPS (DI)(AX*4), Y0          // g
	VMULPS  gegluc<>+0x00(SB), Y0, Y1

	// Held inside [-87, 87]. Below -87 the exponential is nothing; above +87
	// it is more than a float32 exponent field can hold, and the shift that
	// builds 2^n further down would wrap the sign bit and answer with a small
	// number where an enormous one belongs — which reads as g rather than as
	// zero, and is the whole value of the activation instead of none of it.
	VMAXPS  expc<>+0x140(SB), Y1, Y1
	VMINPS  plus87<>+0x00(SB), Y1, Y1

	VMOVUPS Y1, Y2
	VMULPS  expc<>+0x00(SB), Y2, Y2
	VADDPS  expc<>+0x20(SB), Y2, Y2
	VSUBPS  expc<>+0x20(SB), Y2, Y3

	VMOVUPS Y3, Y4
	VMULPS  expc<>+0x40(SB), Y4, Y4
	VSUBPS  Y4, Y1, Y4
	VMOVUPS Y3, Y5
	VMULPS  expc<>+0x60(SB), Y5, Y5
	VSUBPS  Y5, Y4, Y4

	VPSLLD $23, Y2, Y5
	VPADDD expc<>+0x120(SB), Y5, Y5

	VMULPS Y4, Y4, Y6

	VMOVUPS expc<>+0x100(SB), Y7
	VMULPS  Y4, Y7, Y7
	VADDPS  expc<>+0xe0(SB), Y7, Y7
	VMOVUPS expc<>+0xc0(SB), Y8
	VMULPS  Y4, Y8, Y8
	VADDPS  expc<>+0xa0(SB), Y8, Y8
	VMULPS  Y6, Y7, Y7
	VADDPS  Y8, Y7, Y7
	VMULPS  Y6, Y7, Y7
	VMOVUPS expc<>+0x80(SB), Y9
	VMULPS  Y4, Y9, Y9
	VADDPS  Y9, Y7, Y7

	VMULPS Y5, Y7, Y7
	VADDPS Y5, Y7, Y7               // exp(-1.702*g)

	VADDPS  expc<>+0x120(SB), Y7, Y7 // 1 + it
	VDIVPS  Y7, Y0, Y0               // g / it
	VMULPS  (SI)(AX*4), Y0, Y0       // times up
	VMOVUPS Y0, (DI)(AX*4)

	ADDQ $8, AX
	CMPQ AX, CX
	JLT  g8

	VZEROUPPER
	RET

DATA plus87<>+0x00(SB)/4, $0x42ae0000  // +87, where the exponential is all it can be
DATA plus87<>+0x04(SB)/4, $0x42ae0000
DATA plus87<>+0x08(SB)/4, $0x42ae0000
DATA plus87<>+0x0c(SB)/4, $0x42ae0000
DATA plus87<>+0x10(SB)/4, $0x42ae0000
DATA plus87<>+0x14(SB)/4, $0x42ae0000
DATA plus87<>+0x18(SB)/4, $0x42ae0000
DATA plus87<>+0x1c(SB)/4, $0x42ae0000

GLOBL plus87<>(SB), RODATA|NOPTR, $32

DATA negone<>+0x00(SB)/4, $0xbf800000  // -1.0
DATA negone<>+0x04(SB)/4, $0xbf800000
DATA negone<>+0x08(SB)/4, $0xbf800000
DATA negone<>+0x0c(SB)/4, $0xbf800000
DATA negone<>+0x10(SB)/4, $0xbf800000
DATA negone<>+0x14(SB)/4, $0xbf800000
DATA negone<>+0x18(SB)/4, $0xbf800000
DATA negone<>+0x1c(SB)/4, $0xbf800000

GLOBL negone<>(SB), RODATA|NOPTR, $32

// func gatedSigmoidAVX2(dst, num, arg *float32, n int)
//
// dst = num / (1 + exp(-arg)), eight at a time, with the same exponential the
// softmax and the quick GELU above use — ggml's, which is what a conformer
// read out of a GGUF is compared against.
//
// One kernel for two activations, because they are the same expression:
// a SiLU passes the same pointer as num and as arg, and the halving that opens
// a conformer's convolution module passes the two halves of one row. It lives
// in this file rather than beside them because expc<> is file-static, and a
// second copy of ggml's polynomial is a second thing to keep right.
//
// n must be a multiple of eight, and dst may alias num or arg.
TEXT ·gatedSigmoidAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DX
	MOVQ num+8(FP), DI
	MOVQ arg+16(FP), SI
	MOVQ n+24(FP), CX
	XORQ AX, AX

s8:
	VMOVUPS (SI)(AX*4), Y1
	VMULPS  negone<>+0x00(SB), Y1, Y1

	// Held inside [-87, 87], and both ends matter. Below -87 the exponential
	// is nothing; above +87 it is more than a float32 exponent field can
	// hold, and the shift that builds 2^n further down would wrap the sign bit
	// and give a small number where an enormous one belongs. The activation
	// this feeds sees values in the hundreds, so that end is reached.
	VMAXPS  expc<>+0x140(SB), Y1, Y1
	VMINPS  plus87<>+0x00(SB), Y1, Y1

	VMOVUPS Y1, Y2
	VMULPS  expc<>+0x00(SB), Y2, Y2
	VADDPS  expc<>+0x20(SB), Y2, Y2
	VSUBPS  expc<>+0x20(SB), Y2, Y3

	VMOVUPS Y3, Y4
	VMULPS  expc<>+0x40(SB), Y4, Y4
	VSUBPS  Y4, Y1, Y4
	VMOVUPS Y3, Y5
	VMULPS  expc<>+0x60(SB), Y5, Y5
	VSUBPS  Y5, Y4, Y4

	VPSLLD $23, Y2, Y5
	VPADDD expc<>+0x120(SB), Y5, Y5

	VMULPS Y4, Y4, Y6

	VMOVUPS expc<>+0x100(SB), Y7
	VMULPS  Y4, Y7, Y7
	VADDPS  expc<>+0xe0(SB), Y7, Y7
	VMOVUPS expc<>+0xc0(SB), Y8
	VMULPS  Y4, Y8, Y8
	VADDPS  expc<>+0xa0(SB), Y8, Y8
	VMULPS  Y6, Y7, Y7
	VADDPS  Y8, Y7, Y7
	VMULPS  Y6, Y7, Y7
	VMOVUPS expc<>+0x80(SB), Y9
	VMULPS  Y4, Y9, Y9
	VADDPS  Y9, Y7, Y7

	VMULPS Y5, Y7, Y7
	VADDPS Y5, Y7, Y7               // exp(-arg)

	VADDPS  expc<>+0x120(SB), Y7, Y7 // 1 + it
	VMOVUPS (DI)(AX*4), Y0
	VDIVPS  Y7, Y0, Y0
	VMOVUPS Y0, (DX)(AX*4)

	ADDQ $8, AX
	CMPQ AX, CX
	JLT  s8

	VZEROUPPER
	RET
