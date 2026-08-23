//go:build arm64

#include "textflag.h"

// The dot-product instructions, spelled as words.
//
// Go's arm64 assembler has no vector integer multiply at all: no SDOT, no UDOT,
// no SMLAL, not even VMUL. Its SIMD vocabulary is VADD, VFMLA, VTBL, VUSHLL,
// VZIP and a few more, which is enough for the float kernels and nothing like
// enough for the quantized ones.
//
// So they are emitted as constants. The encoding, from the architecture manual:
//
//	bit 31      0
//	bit 30      Q, 1 for the full 128-bit form
//	bit 29      U, 0 signed and 1 unsigned
//	bits 28:24  01110
//	bits 23:22  size, 10 for 4S against 16B
//	bit 21      0
//	bits 20:16  Rm
//	bits 15:12  1001
//	bits 11:10  01
//	bits 9:5    Rn
//	bits 4:0    Rd
//
// which makes SDOT V0.4S, V1.16B, V2.16B into 0x4e829420 and the unsigned one
// 0x6e829420 — one bit apart, which is the whole reason encodings_arm64_test.go
// checks a negative input rather than only a positive one.
//
// These two are not used by any kernel. They exist so that the encodings are a
// tested fact before a kernel depends on them, and so that the next person
// writing one has a worked example rather than a manual.

// func sdotProbe(acc *int32, a, b *int8)
TEXT ·sdotProbe(SB), NOSPLIT, $0-24
	MOVD acc+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	VLD1 (R0), [V0.B16]
	VLD1 (R1), [V1.B16]
	VLD1 (R2), [V2.B16]
	WORD $0x4e829420          // SDOT V0.4S, V1.16B, V2.16B
	VST1 [V0.B16], (R0)
	RET

// func udotProbe(acc *uint32, a, b *uint8)
TEXT ·udotProbe(SB), NOSPLIT, $0-24
	MOVD acc+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	VLD1 (R0), [V0.B16]
	VLD1 (R1), [V1.B16]
	VLD1 (R2), [V2.B16]
	WORD $0x6e829420          // UDOT V0.4S, V1.16B, V2.16B
	VST1 [V0.B16], (R0)
	RET
