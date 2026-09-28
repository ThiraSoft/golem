//go:build amd64

#include "textflag.h"

DATA geluc<>+0(SB)/4, $0.70710678118654752   // 1/sqrt2 (negated on use)
DATA geluc<>+4(SB)/4, $0.5
DATA geluc<>+8(SB)/4, $1.0
DATA geluc<>+12(SB)/4, $-1.26551223
DATA geluc<>+16(SB)/4, $1.00002368
DATA geluc<>+20(SB)/4, $0.37409196
DATA geluc<>+24(SB)/4, $0.09678418
DATA geluc<>+28(SB)/4, $-0.18628806
DATA geluc<>+32(SB)/4, $0.27886807
DATA geluc<>+36(SB)/4, $-1.13520398
DATA geluc<>+40(SB)/4, $1.48851587
DATA geluc<>+44(SB)/4, $-0.82215223
DATA geluc<>+48(SB)/4, $0.17087277
DATA geluc<>+52(SB)/4, $-87.0
DATA geluc<>+56(SB)/4, $1.44269504088896341
DATA geluc<>+60(SB)/4, $0.693359375
DATA geluc<>+64(SB)/4, $-2.12194440e-4
DATA geluc<>+68(SB)/4, $0.0013888888888888889  // 1/720
DATA geluc<>+72(SB)/4, $0.008333333333333333   // 1/120
DATA geluc<>+76(SB)/4, $0.041666666666666664   // 1/24
DATA geluc<>+80(SB)/4, $0.16666666666666666    // 1/6
DATA geluc<>+84(SB)/4, $2.0
DATA geluc<>+88(SB)/4, $0x7fffffff
DATA geluc<>+92(SB)/4, $127
GLOBL geluc<>(SB), RODATA|NOPTR, $96

#define C(i) geluc<>+(i*4)(SB)

// func geluErfAVX2(x *float32, n int)
//
// nn/gelu_erf.go's geluErf1, eight lanes at a time, n a multiple of eight.
// The steps are the same and in the same order, so the kernel and the tail
// agree but for the fused multiply-adds.
TEXT ·geluErfAVX2(SB), NOSPLIT, $0-16
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	SHRQ $3, CX
	JZ   done

loop:
	VMOVUPS (SI), Y0                 // x
	VBROADCASTSS C(0), Y1
	VMULPS Y1, Y0, Y1                // x/sqrt2 = -u
	VBROADCASTSS C(22), Y2
	VANDPS Y2, Y1, Y2                // z = |u|
	// t = 1 / (1 + z/2)
	VBROADCASTSS C(1), Y3
	VBROADCASTSS C(2), Y4
	VFMADD213PS Y4, Y2, Y3           // Y3 = 0.5*z + 1
	VDIVPS Y3, Y4, Y3                // t = 1 / Y3
	// the polynomial in t, from the top
	VBROADCASTSS C(12), Y5
	VBROADCASTSS C(11), Y6
	VFMADD213PS Y6, Y3, Y5
	VBROADCASTSS C(10), Y6
	VFMADD213PS Y6, Y3, Y5
	VBROADCASTSS C(9), Y6
	VFMADD213PS Y6, Y3, Y5
	VBROADCASTSS C(8), Y6
	VFMADD213PS Y6, Y3, Y5
	VBROADCASTSS C(7), Y6
	VFMADD213PS Y6, Y3, Y5
	VBROADCASTSS C(6), Y6
	VFMADD213PS Y6, Y3, Y5
	VBROADCASTSS C(5), Y6
	VFMADD213PS Y6, Y3, Y5
	VBROADCASTSS C(4), Y6
	VFMADD213PS Y6, Y3, Y5
	VMULPS Y3, Y5, Y5                // t*(...)
	VBROADCASTSS C(3), Y6
	VADDPS Y6, Y5, Y5                // -1.26551223 + t*(...)
	VMULPS Y2, Y2, Y6                // z*z
	VSUBPS Y6, Y5, Y5                // e = -z*z - 1.26551223 + t*(...)
	// exp(e), e clamped to -87
	VBROADCASTSS C(13), Y6
	VMAXPS Y6, Y5, Y5
	VBROADCASTSS C(14), Y6
	VMULPS Y6, Y5, Y6                // e*log2e
	VBROADCASTSS C(1), Y7
	VADDPS Y7, Y6, Y6
	VROUNDPS $1, Y6, Y6              // k = floor(e*log2e + 0.5)
	VBROADCASTSS C(15), Y7
	VMULPS Y7, Y6, Y7
	VSUBPS Y7, Y5, Y5                // e - k*ln2hi
	VBROADCASTSS C(16), Y7
	VMULPS Y7, Y6, Y7
	VSUBPS Y7, Y5, Y5                // r = e - k*ln2hi - k*ln2lo
	VBROADCASTSS C(17), Y7
	VBROADCASTSS C(18), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(19), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(20), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(1), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(2), Y8
	VFMADD213PS Y8, Y5, Y7
	VFMADD213PS Y8, Y5, Y7           // p
	VCVTPS2DQ Y6, Y6                 // k as integers (exact: k is whole)
	VPBROADCASTD C(23), Y8
	VPADDD Y8, Y6, Y6
	VPSLLD $23, Y6, Y6               // 2^k
	VMULPS Y6, Y7, Y7
	VMULPS Y7, Y3, Y7                // r = t*exp(e) = erfc(|u|)
	// erfc(u) = u >= 0 ? r : 2 - r, and u < 0 exactly when x > 0
	VBROADCASTSS C(21), Y8
	VSUBPS Y7, Y8, Y8                // 2 - r
	VXORPS Y9, Y9, Y9
	VCMPPS $0x1E, Y9, Y0, Y9         // x > 0
	VBLENDVPS Y9, Y8, Y7, Y7
	VBROADCASTSS C(1), Y8
	VMULPS Y8, Y0, Y0
	VMULPS Y7, Y0, Y0                // 0.5*x*erfc(u)
	VMOVUPS Y0, (SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  loop
	VZEROUPPER

done:
	RET

// func expSubAVX2(x *float32, n int, m float32)
//
// x[i] = e^(x[i]-m) for n a multiple of eight, by expF32's steps; what falls
// below e^-87, a masked score among them, is zero.
TEXT ·expSubAVX2(SB), NOSPLIT, $0-20
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	VBROADCASTSS m+16(FP), Y10
	VBROADCASTSS C(13), Y11          // -87
	SHRQ $3, CX
	JZ   expdone

exploop:
	VMOVUPS (SI), Y5
	VSUBPS Y10, Y5, Y5               // v = x - m
	VCMPPS $0x1D, Y11, Y5, Y9        // v >= -87
	VMAXPS Y11, Y5, Y5
	VBROADCASTSS C(14), Y6
	VMULPS Y6, Y5, Y6
	VBROADCASTSS C(1), Y7
	VADDPS Y7, Y6, Y6
	VROUNDPS $1, Y6, Y6              // k
	VBROADCASTSS C(15), Y7
	VMULPS Y7, Y6, Y7
	VSUBPS Y7, Y5, Y5
	VBROADCASTSS C(16), Y7
	VMULPS Y7, Y6, Y7
	VSUBPS Y7, Y5, Y5                // r
	VBROADCASTSS C(17), Y7
	VBROADCASTSS C(18), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(19), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(20), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(1), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(2), Y8
	VFMADD213PS Y8, Y5, Y7
	VFMADD213PS Y8, Y5, Y7
	VCVTPS2DQ Y6, Y6
	VPBROADCASTD C(23), Y8
	VPADDD Y8, Y6, Y6
	VPSLLD $23, Y6, Y6
	VMULPS Y6, Y7, Y7
	VANDPS Y9, Y7, Y7                // zero below the range
	VMOVUPS Y7, (SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  exploop
	VZEROUPPER

expdone:
	RET

// func softmaxF32AVX2(x *float32, n int)
//
// SoftmaxF32 for n a multiple of eight: the maximum, e^(x-max) by expF32's
// steps with what falls below e^-87 zero, their sum, and each divided by it.
TEXT ·softmaxF32AVX2(SB), NOSPLIT, $0-16
	MOVQ x+0(FP), DI
	MOVQ n+8(FP), DX
	SHRQ $3, DX
	JZ   smdone

	// The maximum.
	MOVQ DI, SI
	MOVQ DX, CX
	VMOVUPS (SI), Y10
smmax:
	VMAXPS (SI), Y10, Y10
	ADDQ $32, SI
	DECQ CX
	JNZ  smmax
	VPERM2F128 $1, Y10, Y10, Y0
	VMAXPS Y0, Y10, Y10
	VPERMILPS $0x4E, Y10, Y0
	VMAXPS Y0, Y10, Y10
	VPERMILPS $0xB1, Y10, Y0
	VMAXPS Y0, Y10, Y10             // every lane the maximum

	VBROADCASTSS C(13), Y11          // -87
	VXORPS Y4, Y4, Y4                // the sum
	MOVQ DI, SI
	MOVQ DX, CX
smexp:
	VMOVUPS (SI), Y5
	VSUBPS Y10, Y5, Y5
	VCMPPS $0x1D, Y11, Y5, Y9
	VMAXPS Y11, Y5, Y5
	VBROADCASTSS C(14), Y6
	VMULPS Y6, Y5, Y6
	VBROADCASTSS C(1), Y7
	VADDPS Y7, Y6, Y6
	VROUNDPS $1, Y6, Y6
	VBROADCASTSS C(15), Y7
	VMULPS Y7, Y6, Y7
	VSUBPS Y7, Y5, Y5
	VBROADCASTSS C(16), Y7
	VMULPS Y7, Y6, Y7
	VSUBPS Y7, Y5, Y5
	VBROADCASTSS C(17), Y7
	VBROADCASTSS C(18), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(19), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(20), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(1), Y8
	VFMADD213PS Y8, Y5, Y7
	VBROADCASTSS C(2), Y8
	VFMADD213PS Y8, Y5, Y7
	VFMADD213PS Y8, Y5, Y7
	VCVTPS2DQ Y6, Y6
	VPBROADCASTD C(23), Y8
	VPADDD Y8, Y6, Y6
	VPSLLD $23, Y6, Y6
	VMULPS Y6, Y7, Y7
	VANDPS Y9, Y7, Y7
	VADDPS Y7, Y4, Y4
	VMOVUPS Y7, (SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  smexp
	VPERM2F128 $1, Y4, Y4, Y0
	VADDPS Y0, Y4, Y4
	VPERMILPS $0x4E, Y4, Y0
	VADDPS Y0, Y4, Y4
	VPERMILPS $0xB1, Y4, Y0
	VADDPS Y0, Y4, Y4                // every lane the sum

	MOVQ DI, SI
	MOVQ DX, CX
smdiv:
	VMOVUPS (SI), Y5
	VDIVPS Y4, Y5, Y5
	VMOVUPS Y5, (SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  smdiv
	VZEROUPPER

smdone:
	RET
