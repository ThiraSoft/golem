package vk

// Talking Head(?) Anime 3 on the card: the convolutions and the norm.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_dw.comp -o shaders/tha3_dw.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_dwt.comp -o shaders/tha3_dwt.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_pw.comp -o shaders/tha3_pw.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_sum.comp -o shaders/tha3_sum.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_head.comp -o shaders/tha3_head.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_stats.comp -o shaders/tha3_stats.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_norm.comp -o shaders/tha3_norm.spv

//go:embed shaders/tha3_dw.spv
var tha3DWSPIRV []byte

//go:embed shaders/tha3_dwt.spv
var tha3DWTSPIRV []byte

//go:embed shaders/tha3_pw.spv
var tha3PWSPIRV []byte

//go:embed shaders/tha3_sum.spv
var tha3SumSPIRV []byte

//go:embed shaders/tha3_head.spv
var tha3HeadSPIRV []byte

//go:embed shaders/tha3_stats.spv
var tha3StatsSPIRV []byte

//go:embed shaders/tha3_norm.spv
var tha3NormSPIRV []byte

// tha3SliceLen is how many pixels of a channel one statistics workgroup
// takes. A 512×512 plane is 64 slices, which is what keeps the editor's
// 32-channel norms from running on 32 workgroups of a 64-unit card.
const tha3SliceLen = 4096

type tha3DWPush struct{ src, dst, h, w, oh, ow, k, stride, pad, weights uint32 }

type tha3DWTPush struct{ src, dst, h, w, oh, ow, weights uint32 }

type tha3PWPush struct{ src, dst, cin, cout, pix, weights, bias, hasBias, slices, kper uint32 }

type tha3SumPush struct{ src, dst, cout, pix, slices, bias, hasBias uint32 }

type tha3HeadPush struct{ src, dst, cin, cout, h, w, weights, bias, hasBias, act uint32 }

type tha3StatsPush struct{ src, plane, sliceLen, slices, stats uint32 }

type tha3NormPush struct{ x, plane, slices, stats, gamma, beta, act, residual, hasResidual uint32 }

func init() {
	tha3Kernels[tha3DW] = tha3KernelSpec{tha3DWSPIRV, unsafe.Sizeof(tha3DWPush{})}
	tha3Kernels[tha3DWT] = tha3KernelSpec{tha3DWTSPIRV, unsafe.Sizeof(tha3DWTPush{})}
	tha3Kernels[tha3PW] = tha3KernelSpec{tha3PWSPIRV, unsafe.Sizeof(tha3PWPush{})}
	tha3Kernels[tha3Sum] = tha3KernelSpec{tha3SumSPIRV, unsafe.Sizeof(tha3SumPush{})}
	tha3Kernels[tha3Head] = tha3KernelSpec{tha3HeadSPIRV, unsafe.Sizeof(tha3HeadPush{})}
	tha3Kernels[tha3Stats] = tha3KernelSpec{tha3StatsSPIRV, unsafe.Sizeof(tha3StatsPush{})}
	tha3Kernels[tha3Norm] = tha3KernelSpec{tha3NormSPIRV, unsafe.Sizeof(tha3NormPush{})}
}

func need(what string, got, want int) {
	if got != want {
		panic(fmt.Sprintf("vk: THA3 %s has %d floats, want %d", what, got, want))
	}
}

// DW is a depthwise k×k convolution without bias; w is [C][1][k][k].
func (g *THA3Graph) DW(x THA3Tensor, w THA3Weights, k, stride, pad int) THA3Tensor {
	need("depthwise weight", w.n, x.C*k*k)
	oh, ow := (x.H+2*pad-k)/stride+1, (x.W+2*pad-k)/stride+1
	out := g.tensor(x.C, oh, ow)
	g.op([]THA3Tensor{x}, []THA3Tensor{out}, func(r *Recorder, rx *THA3Runner) {
		push := tha3DWPush{src: rx.at(x), dst: rx.at(out), h: uint32(x.H), w: uint32(x.W), oh: uint32(oh), ow: uint32(ow),
			k: uint32(k), stride: uint32(stride), pad: uint32(pad), weights: uint32(w.off)}
		rx.dispatch(r, tha3DW, groups256(oh*ow), x.C, unsafe.Pointer(&push))
	})
	return out
}

// DWT is the depthwise transposed 4×4, stride 2, padding 1; w is [C][1][4][4].
func (g *THA3Graph) DWT(x THA3Tensor, w THA3Weights) THA3Tensor {
	need("transposed weight", w.n, x.C*16)
	out := g.tensor(x.C, 2*x.H, 2*x.W)
	g.op([]THA3Tensor{x}, []THA3Tensor{out}, func(r *Recorder, rx *THA3Runner) {
		push := tha3DWTPush{src: rx.at(x), dst: rx.at(out), h: uint32(x.H), w: uint32(x.W),
			oh: uint32(out.H), ow: uint32(out.W), weights: uint32(w.off)}
		rx.dispatch(r, tha3DWT, groups256(out.H*out.W), x.C, unsafe.Pointer(&push))
	})
	return out
}

// PW is a 1×1 convolution to out channels; w is [out][C], bias has out floats
// or none.
func (g *THA3Graph) PW(x THA3Tensor, w, bias THA3Weights, out int) THA3Tensor {
	need("pointwise weight", w.n, out*x.C)
	if bias.n != 0 {
		need("pointwise bias", bias.n, out)
	}
	y := g.tensor(out, x.H, x.W)
	pix := x.H * x.W
	pixTiles, outTiles := (pix+63)/64, (out+63)/64
	slices, kper := pwSlices(pixTiles*outTiles, x.C)
	if slices == 1 {
		g.op([]THA3Tensor{x}, []THA3Tensor{y}, func(r *Recorder, rx *THA3Runner) {
			push := tha3PWPush{src: rx.at(x), dst: rx.at(y), cin: uint32(x.C), cout: uint32(out), pix: uint32(pix),
				weights: uint32(w.off), slices: 1, kper: uint32(x.C)}
			if bias.n != 0 {
				push.bias, push.hasBias = uint32(bias.off), 1
			}
			rx.dispatch(r, tha3PW, pixTiles, outTiles, unsafe.Pointer(&push))
		})
		return y
	}
	parts := g.tensor(slices*out, x.H, x.W)
	g.op([]THA3Tensor{x}, []THA3Tensor{parts}, func(r *Recorder, rx *THA3Runner) {
		push := tha3PWPush{src: rx.at(x), dst: rx.at(parts), cin: uint32(x.C), cout: uint32(out), pix: uint32(pix),
			weights: uint32(w.off), slices: uint32(slices), kper: uint32(kper)}
		rx.dispatch(r, tha3PW, pixTiles*slices, outTiles, unsafe.Pointer(&push))
	})
	g.op([]THA3Tensor{parts}, []THA3Tensor{y}, func(r *Recorder, rx *THA3Runner) {
		push := tha3SumPush{src: rx.at(parts), dst: rx.at(y), cout: uint32(out), pix: uint32(pix), slices: uint32(slices)}
		if bias.n != 0 {
			push.bias, push.hasBias = uint32(bias.off), 1
		}
		rx.dispatch(r, tha3Sum, groups256(pix), out, unsafe.Pointer(&push))
	})
	return y
}

// pwTarget is how many workgroups a product is split to reach, and pwMinK the
// fewest input channels a slice walks: below that the sum pass costs more than
// the slice saves. Measured on the RX 9070 XT, the whole pose went from 24.2 to
// 22 ms; 128 to 512 workgroups all land within the noise of it, and slices
// of 32 channels are slower.
const pwTarget, pwMinK = 256, 64

// pwSlices splits a product of tiles tiles over cin input channels into
// slices of kper channels, a whole number of the kernel's sixteen.
func pwSlices(tiles, cin int) (slices, kper int) {
	slices = min((pwTarget+tiles-1)/tiles, cin/pwMinK)
	if slices <= 1 {
		return 1, cin
	}
	kper = (cin + slices - 1) / slices
	kper = (kper + 15) / 16 * 16
	return (cin + kper - 1) / kper, kper
}

// Head is a dense 3×3 to at most four channels, a bias or none, and a sigmoid,
// a tanh or nothing.
func (g *THA3Graph) Head(x THA3Tensor, w, bias THA3Weights, out int, act THA3Act) THA3Tensor {
	if out < 1 || out > 4 {
		panic(fmt.Sprintf("vk: THA3 head of %d outputs", out))
	}
	if act != THA3None && act != THA3Sigmoid && act != THA3Tanh {
		panic(fmt.Sprintf("vk: THA3 head with activation %d", act))
	}
	need("head weight", w.n, out*x.C*9)
	if bias.n != 0 {
		need("head bias", bias.n, out)
	}
	y := g.tensor(out, x.H, x.W)
	g.op([]THA3Tensor{x}, []THA3Tensor{y}, func(r *Recorder, rx *THA3Runner) {
		push := tha3HeadPush{src: rx.at(x), dst: rx.at(y), cin: uint32(x.C), cout: uint32(out), h: uint32(x.H), w: uint32(x.W),
			weights: uint32(w.off), act: uint32(act)}
		if bias.n != 0 {
			push.bias, push.hasBias = uint32(bias.off), 1
		}
		rx.dispatch(r, tha3Head, groups256(x.H*x.W), 1, unsafe.Pointer(&push))
	})
	return y
}

// Norm is InstanceNorm2d(affine) on x in place, then act, then residual added
// if there is one. It returns x.
func (g *THA3Graph) Norm(x THA3Tensor, gamma, beta THA3Weights, act THA3Act, residual *THA3Tensor) THA3Tensor {
	need("norm weight", gamma.n, x.C)
	need("norm bias", beta.n, x.C)
	if act != THA3None && act != THA3ReLU && act != THA3Leaky {
		panic(fmt.Sprintf("vk: THA3 norm with activation %d", act))
	}
	plane := x.H * x.W
	slices := (plane + tha3SliceLen - 1) / tha3SliceLen
	stats := g.tensor(1, 1, x.C*slices*3)
	reads := []THA3Tensor{x}
	if residual != nil {
		if residual.C != x.C || residual.H != x.H || residual.W != x.W {
			panic("vk: THA3 residual of another shape")
		}
		reads = append(reads, *residual)
	}
	g.op(reads, []THA3Tensor{stats}, func(r *Recorder, rx *THA3Runner) {
		sp := tha3StatsPush{src: rx.at(x), plane: uint32(plane), sliceLen: tha3SliceLen, slices: uint32(slices), stats: rx.at(stats)}
		rx.dispatch(r, tha3Stats, slices, x.C, unsafe.Pointer(&sp))
		r.Barrier()
		np := tha3NormPush{x: rx.at(x), plane: uint32(plane), slices: uint32(slices), stats: rx.at(stats),
			gamma: uint32(gamma.off), beta: uint32(beta.off), act: uint32(act)}
		if residual != nil {
			np.residual, np.hasResidual = rx.at(*residual), 1
		}
		rx.dispatch(r, tha3Norm, groups256(plane), x.C, unsafe.Pointer(&np))
	})
	return x
}
