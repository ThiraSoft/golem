package vk

// Talking Head(?) Anime 3 on the card: the eye upscaler, a chain of dense 3×3
// convolutions with PReLUs, and the step that lays what it makes over the
// larger picture's face.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_conv3.comp -o shaders/tha3_conv3.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_upscale.comp -o shaders/tha3_upscale.spv

//go:embed shaders/tha3_conv3.spv
var tha3Conv3SPIRV []byte

//go:embed shaders/tha3_upscale.spv
var tha3UpscaleSPIRV []byte

type tha3Conv3Push struct{ src, dst, cin, cout, h, w, weights, bias, slopes, hasSlopes uint32 }

type tha3UpscalePush struct {
	dst, face, mask, feat, lr, lh, lw, h, w uint32
	amount                                  float32
}

func init() {
	tha3Kernels[tha3Conv3] = tha3KernelSpec{tha3Conv3SPIRV, unsafe.Sizeof(tha3Conv3Push{})}
	tha3Kernels[tha3Upscale] = tha3KernelSpec{tha3UpscaleSPIRV, unsafe.Sizeof(tha3UpscalePush{})}
}

// Conv3 is a dense 3×3 convolution, stride 1, padding 1, to out channels,
// with a bias, and a PReLU when slopes holds out floats; w is [out][C][3][3].
func (g *THA3Graph) Conv3(x THA3Tensor, w, bias, slopes THA3Weights, out int) THA3Tensor {
	need("3x3 weight", w.n, out*x.C*9)
	need("3x3 bias", bias.n, out)
	if slopes.n != 0 {
		need("PReLU slopes", slopes.n, out)
	}
	y := g.tensor(out, x.H, x.W)
	g.op([]THA3Tensor{x}, []THA3Tensor{y}, func(r *Recorder, rx *THA3Runner) {
		push := tha3Conv3Push{src: rx.at(x), dst: rx.at(y), cin: uint32(x.C), cout: uint32(out), h: uint32(x.H), w: uint32(x.W),
			weights: uint32(w.off), bias: uint32(bias.off)}
		if slopes.n != 0 {
			push.slopes, push.hasSlopes = uint32(slopes.off), 1
		}
		rx.dispatch(r, tha3Conv3, (x.H*x.W+63)/64, (out+63)/64, unsafe.Pointer(&push))
	})
	return y
}

// ToSRGB is the colour of a picture, linear in [-1, 1], as sRGB in [0, 1]:
// three channels.
func (g *THA3Graph) ToSRGB(image THA3Tensor) THA3Tensor {
	if image.C < 3 {
		panic(fmt.Sprintf("vk: THA3 sRGB of %d channels", image.C))
	}
	out := g.tensor(3, image.H, image.W)
	g.op([]THA3Tensor{image}, []THA3Tensor{out}, func(r *Recorder, x *THA3Runner) {
		push := tha3BlendPush{dst: x.at(out), image: x.at(image), change: x.at(image), alpha: x.at(image),
			plane: uint32(image.H * image.W), alphaC: 1, mode: 6}
		x.dispatch(r, tha3Blend, groups256(image.H*image.W), 3, unsafe.Pointer(&push))
	})
	return out
}

// Spread is mask multiplied by gain and clamped to one.
func (g *THA3Graph) Spread(mask THA3Tensor, gain float32) THA3Tensor {
	return g.blendBy(7, mask, mask, mask, gain)
}

// UpscaleLay lays the upscaler's output over face, a picture of four
// channels, through mask, of face's size, by amount: feat is the last
// convolution's 48 channels and lr the sRGB picture it was run on, which
// together make the picture four times larger. face's alpha is kept.
func (g *THA3Graph) UpscaleLay(face, mask, feat, lr THA3Tensor, amount float32) THA3Tensor {
	if face.C != 4 || mask.C != 1 || mask.H != face.H || mask.W != face.W || feat.C != 48 || lr.C != 3 ||
		feat.H != lr.H || feat.W != lr.W {
		panic("vk: THA3 upscale of mismatched shapes")
	}
	out := g.tensor(4, face.H, face.W)
	g.op([]THA3Tensor{face, mask, feat, lr}, []THA3Tensor{out}, func(r *Recorder, x *THA3Runner) {
		push := tha3UpscalePush{dst: x.at(out), face: x.at(face), mask: x.at(mask), feat: x.at(feat), lr: x.at(lr),
			lh: uint32(lr.H), lw: uint32(lr.W), h: uint32(face.H), w: uint32(face.W), amount: amount}
		x.dispatch(r, tha3Upscale, groups256(face.H*face.W), 4, unsafe.Pointer(&push))
	})
	return out
}
