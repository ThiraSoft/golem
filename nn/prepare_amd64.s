//go:build amd64

#include "textflag.h"

DATA prepc<>+0x00(SB)/4, $0x00007fff  // the round-to-nearest bias of a bfloat16
DATA prepc<>+0x04(SB)/4, $0x00007fff
DATA prepc<>+0x08(SB)/4, $0x00007fff
DATA prepc<>+0x0c(SB)/4, $0x00007fff
DATA prepc<>+0x10(SB)/4, $0x00007fff
DATA prepc<>+0x14(SB)/4, $0x00007fff
DATA prepc<>+0x18(SB)/4, $0x00007fff
DATA prepc<>+0x1c(SB)/4, $0x00007fff

DATA prepc<>+0x20(SB)/4, $0x00000001  // the tie-breaking bit
DATA prepc<>+0x24(SB)/4, $0x00000001
DATA prepc<>+0x28(SB)/4, $0x00000001
DATA prepc<>+0x2c(SB)/4, $0x00000001
DATA prepc<>+0x30(SB)/4, $0x00000001
DATA prepc<>+0x34(SB)/4, $0x00000001
DATA prepc<>+0x38(SB)/4, $0x00000001
DATA prepc<>+0x3c(SB)/4, $0x00000001

DATA prepc<>+0x40(SB)/4, $0xffff0000  // what a bfloat16 keeps
DATA prepc<>+0x44(SB)/4, $0xffff0000
DATA prepc<>+0x48(SB)/4, $0xffff0000
DATA prepc<>+0x4c(SB)/4, $0xffff0000
DATA prepc<>+0x50(SB)/4, $0xffff0000
DATA prepc<>+0x54(SB)/4, $0xffff0000
DATA prepc<>+0x58(SB)/4, $0xffff0000
DATA prepc<>+0x5c(SB)/4, $0xffff0000

GLOBL prepc<>(SB), RODATA|NOPTR, $96

// func prepareIlvAVX2(dst, src *float32, n int, lo, hi float32)
//
// The activation of one projection, ready for the interleaved kernel: held
// inside the range its weights were fitted for, rounded to bfloat16, and
// written in the order interleaving those weights with zeros produces.
//
// All three in one sweep, eight values at a time. Scalar, this was five
// percent of the tower — it runs once per projection over the whole grid, and
// a rounding that is four integer operations in a register was costing a call
// and a branch.
//
// n must be a multiple of sixteen.
TEXT ·prepareIlvAVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	VBROADCASTSS lo+24(FP), Y14
	VBROADCASTSS hi+28(FP), Y15

	XORQ AX, AX

p16:
	VMOVUPS (SI)(AX*4), Y0
	VMOVUPS 32(SI)(AX*4), Y1

	VMAXPS Y14, Y0, Y0
	VMINPS Y15, Y0, Y0
	VMAXPS Y14, Y1, Y1
	VMINPS Y15, Y1, Y1

	// Round to nearest, ties to even: add half a bfloat16's last place plus
	// the bit that decides the tie, then drop what does not fit.
	VPSRLD $16, Y0, Y2
	VPAND  prepc<>+0x20(SB), Y2, Y2
	VPADDD prepc<>+0x00(SB), Y0, Y0
	VPADDD Y2, Y0, Y0
	VPAND  prepc<>+0x40(SB), Y0, Y0

	VPSRLD $16, Y1, Y3
	VPAND  prepc<>+0x20(SB), Y3, Y3
	VPADDD prepc<>+0x00(SB), Y1, Y1
	VPADDD Y3, Y1, Y1
	VPAND  prepc<>+0x40(SB), Y1, Y1

	// And the order: the first four and the ninth to twelfth, then the rest.
	VPERM2F128 $0x20, Y1, Y0, Y2
	VPERM2F128 $0x31, Y1, Y0, Y3
	VMOVUPS    Y2, (DI)(AX*4)
	VMOVUPS    Y3, 32(DI)(AX*4)

	ADDQ $16, AX
	CMPQ AX, CX
	JLT  p16

	VZEROUPPER
	RET

// func clampAVX2(x *float32, n int, lo, hi float32)
//
// Every value held inside [lo, hi], eight at a time. It is what a
// quantization-aware weight's output range costs, and it was the last scalar
// loop in the tower's hot path.
TEXT ·clampAVX2(SB), NOSPLIT, $0-24
	MOVQ x+0(FP), DI
	MOVQ n+8(FP), CX
	VBROADCASTSS lo+16(FP), Y14
	VBROADCASTSS hi+20(FP), Y15
	XORQ AX, AX

	MOVQ CX, DX
	ANDQ $-8, DX
	JMP  c8cond

c8:
	VMOVUPS (DI)(AX*4), Y0
	VMAXPS  Y14, Y0, Y0
	VMINPS  Y15, Y0, Y0
	VMOVUPS Y0, (DI)(AX*4)
	ADDQ    $8, AX

c8cond:
	CMPQ AX, DX
	JLT  c8

c1:
	CMPQ   AX, CX
	JGE    done
	VMOVSS (DI)(AX*4), X0
	VMAXSS X14, X0, X0
	VMINSS X15, X0, X0
	VMOVSS X0, (DI)(AX*4)
	INCQ   AX
	JMP    c1

done:
	VZEROUPPER
	RET
