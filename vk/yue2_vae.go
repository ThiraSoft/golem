package vk

// YuE2's Oobleck decoder on the Krea 2 machine: the two kernels its
// convolutions need beside the products. yue2/ lays the decoder out.

import (
	_ "embed"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/yue2_im2col.comp -o shaders/yue2_im2col.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/yue2_col2im.comp -o shaders/yue2_col2im.spv

//go:embed shaders/yue2_im2col.spv
var yue2Im2ColSPIRV []byte

//go:embed shaders/yue2_col2im.spv
var yue2Col2ImSPIRV []byte

// K2Yue2Im2Col is the operand of a convolution of K taps, dilated by Dil and
// padded by Pad, over a time-major activation of IC channels: X[t][ic*K+k]
// is the input at t + k*Dil - Pad through SnakeBeta, or zero outside. See
// shaders/yue2_im2col.comp.
type K2Yue2Im2Col struct {
	Src            uint32 // floats, [t][ic]
	Dst, DstStride uint32 // halves with Halves set, floats without
	T, IC          uint32
	K, Dil, Pad    uint32
	// exp(alpha) and 1/exp(beta) in the parameters; K2None for no snake.
	SnakeA, SnakeB uint32
	Halves         uint32
}

// K2Yue2Col2Im gathers a transposed convolution's product, Col[t][oc*K+k],
// into Dst[t][oc] plus a bias (K2None for none). See shaders/yue2_col2im.comp.
type K2Yue2Col2Im struct {
	Col, Dst    uint32
	TIn, TOut   uint32
	OC, K       uint32
	Stride, Pad uint32
	Bias        uint32
}

func (k *K2) Yue2Im2Col(r *Recorder, p K2Yue2Im2Col) {
	k.dispatch(r, k2Yue2Im2Col, -1, groups256(int(p.T+2*p.Pad)*int(p.IC)), unsafe.Pointer(&p))
}

func (k *K2) Yue2Col2Im(r *Recorder, p K2Yue2Col2Im) {
	k.dispatch(r, k2Yue2Col2Im, -1, groups256(int(p.TOut)*int(p.OC)), unsafe.Pointer(&p))
}
