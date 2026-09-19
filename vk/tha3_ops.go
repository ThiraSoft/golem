package vk

// Talking Head(?) Anime 3 on the card: the operations that move floats,
// blend them or resample them.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_copy.comp -o shaders/tha3_copy.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_pose.comp -o shaders/tha3_pose.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_blend.comp -o shaders/tha3_blend.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_resize.comp -o shaders/tha3_resize.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/tha3_warp.comp -o shaders/tha3_warp.spv

//go:embed shaders/tha3_copy.spv
var tha3CopySPIRV []byte

//go:embed shaders/tha3_pose.spv
var tha3PoseSPIRV []byte

//go:embed shaders/tha3_blend.spv
var tha3BlendSPIRV []byte

//go:embed shaders/tha3_resize.spv
var tha3ResizeSPIRV []byte

//go:embed shaders/tha3_warp.spv
var tha3WarpSPIRV []byte

type tha3CopyPush struct {
	src, srcW, srcY, srcX, srcPlane uint32
	dst, dstW, dstY, dstX, dstPlane uint32
	h, w, up                        uint32
}

type tha3PosePush struct{ dst, plane, from uint32 }

type tha3BlendPush struct{ dst, image, change, alpha, plane, alphaC, mode uint32 }

type tha3ResizePush struct{ src, srcH, srcW, dst, dstH, dstW uint32 }

type tha3WarpPush struct{ image, grid, add, hasAdd, dst, h, w, c uint32 }

func init() {
	tha3Kernels[tha3Copy] = tha3KernelSpec{tha3CopySPIRV, unsafe.Sizeof(tha3CopyPush{})}
	tha3Kernels[tha3Pose] = tha3KernelSpec{tha3PoseSPIRV, unsafe.Sizeof(tha3PosePush{})}
	tha3Kernels[tha3Blend] = tha3KernelSpec{tha3BlendSPIRV, unsafe.Sizeof(tha3BlendPush{})}
	tha3Kernels[tha3Resize] = tha3KernelSpec{tha3ResizeSPIRV, unsafe.Sizeof(tha3ResizePush{})}
	tha3Kernels[tha3Warp] = tha3KernelSpec{tha3WarpSPIRV, unsafe.Sizeof(tha3WarpPush{})}
}

// copyInto copies an h×w rectangle of every channel of src, starting at
// (sy, sx), to dst at (dy, dx); with up at 2 the source is read at half the
// coordinates. creates says whether this op is the one that makes dst.
func (g *THA3Graph) copyInto(dst THA3Tensor, dy, dx int, src THA3Tensor, sy, sx, h, w, up int, creates bool) {
	reads := []THA3Tensor{src}
	var made []THA3Tensor
	if creates {
		made = []THA3Tensor{dst}
	} else {
		reads = append(reads, dst)
	}
	g.op(reads, made, func(r *Recorder, x *THA3Runner) {
		push := tha3CopyPush{
			src: x.at(src), srcW: uint32(src.W), srcY: uint32(sy), srcX: uint32(sx), srcPlane: uint32(src.H * src.W),
			dst: x.at(dst), dstW: uint32(dst.W), dstY: uint32(dy), dstX: uint32(dx), dstPlane: uint32(dst.H * dst.W),
			h: uint32(h), w: uint32(w), up: uint32(up),
		}
		x.dispatch(r, tha3Copy, groups256(h*w), src.C, unsafe.Pointer(&push))
	})
}

// Crop is image[:, y0:y0+h, x0:x0+w].
func (g *THA3Graph) Crop(x THA3Tensor, y0, x0, h, w int) THA3Tensor {
	out := g.tensor(x.C, h, w)
	g.copyInto(out, 0, 0, x, y0, x0, h, w, 1, true)
	return out
}

// Paste is a copy of dst with src written at (y0, x0). dst itself is left as
// it was, which is what lets the pinned picture be pasted into every pose.
func (g *THA3Graph) Paste(dst, src THA3Tensor, y0, x0 int) THA3Tensor {
	if src.C != dst.C {
		panic(fmt.Sprintf("vk: pasting %d channels into %d", src.C, dst.C))
	}
	out := g.tensor(dst.C, dst.H, dst.W)
	g.copyInto(out, 0, 0, dst, 0, 0, dst.H, dst.W, 1, true)
	g.copyInto(out, y0, x0, src, 0, 0, src.H, src.W, 1, false)
	return out
}

// Concat stacks tensors of one size along the channels.
func (g *THA3Graph) Concat(parts ...THA3Tensor) THA3Tensor {
	c := 0
	for _, p := range parts {
		if p.H != parts[0].H || p.W != parts[0].W {
			panic(fmt.Sprintf("vk: concatenating %dx%d with %dx%d", p.H, p.W, parts[0].H, parts[0].W))
		}
		c += p.C
	}
	out := g.tensor(c, parts[0].H, parts[0].W)
	at := 0
	for i, p := range parts {
		g.copyInto(out.Channels(at, at+p.C), 0, 0, p, 0, 0, p.H, p.W, 1, i == 0)
		at += p.C
	}
	return out
}

// Upsample2 doubles both sides by nearest neighbour.
func (g *THA3Graph) Upsample2(x THA3Tensor) THA3Tensor {
	out := g.tensor(x.C, 2*x.H, 2*x.W)
	g.copyInto(out, 0, 0, x, 0, 0, 2*x.H, 2*x.W, 2, true)
	return out
}

// PoseSlice is pose[from:from+n] tiled over h×w planes, read from the pose
// buffer when the pass runs.
func (g *THA3Graph) PoseSlice(from, n, h, w int) THA3Tensor {
	out := g.tensor(n, h, w)
	g.op(nil, []THA3Tensor{out}, func(r *Recorder, x *THA3Runner) {
		push := tha3PosePush{dst: x.at(out), plane: uint32(h * w), from: uint32(from)}
		x.dispatch(r, tha3Pose, groups256(h*w), n, unsafe.Pointer(&push))
	})
	return out
}

func (g *THA3Graph) blend(mode uint32, alpha, change, image THA3Tensor) THA3Tensor {
	out := g.tensor(image.C, image.H, image.W)
	g.op([]THA3Tensor{alpha, change, image}, []THA3Tensor{out}, func(r *Recorder, x *THA3Runner) {
		push := tha3BlendPush{
			dst: x.at(out), image: x.at(image), change: x.at(change), alpha: x.at(alpha),
			plane: uint32(image.H * image.W), alphaC: uint32(alpha.C), mode: mode,
		}
		x.dispatch(r, tha3Blend, groups256(image.H*image.W), image.C, unsafe.Pointer(&push))
	})
	return out
}

// ColorChange is tha3's applyColorChange: change*alpha + image*(1-alpha).
func (g *THA3Graph) ColorChange(alpha, change, image THA3Tensor) THA3Tensor {
	return g.blend(0, alpha, change, image)
}

// RGBChange is applyRGBChange: the colour channels blended, image's alpha kept.
func (g *THA3Graph) RGBChange(alpha, change, image THA3Tensor) THA3Tensor {
	return g.blend(1, alpha, change, image)
}

// RGBHalfAlpha is RGBChange with alpha = (change's fourth channel + 1) / 2,
// the eyebrow combiner's output 2.
func (g *THA3Graph) RGBHalfAlpha(change, image THA3Tensor) THA3Tensor {
	// Mode 2 reads its alpha from change itself; the alpha handle is change
	// again only so the op has one.
	return g.blend(2, change, change, image)
}

// Resize is bilinear, align_corners=False, no antialias.
func (g *THA3Graph) Resize(x THA3Tensor, h, w int) THA3Tensor {
	out := g.tensor(x.C, h, w)
	g.op([]THA3Tensor{x}, []THA3Tensor{out}, func(r *Recorder, rx *THA3Runner) {
		push := tha3ResizePush{src: rx.at(x), srcH: uint32(x.H), srcW: uint32(x.W), dst: rx.at(out), dstH: uint32(h), dstW: uint32(w)}
		rx.dispatch(r, tha3Resize, groups256(h*w), x.C, unsafe.Pointer(&push))
	})
	return out
}

// Warp is applyGridChange: image sampled at the identity grid plus grid.
func (g *THA3Graph) Warp(image, grid THA3Tensor) THA3Tensor { return g.warp(image, grid, nil) }

// WarpAdd is Warp over grid + add, the editor's sum of its own grid and the
// rotator's.
func (g *THA3Graph) WarpAdd(image, grid, add THA3Tensor) THA3Tensor {
	return g.warp(image, grid, &add)
}

func (g *THA3Graph) warp(image, grid THA3Tensor, add *THA3Tensor) THA3Tensor {
	out := g.tensor(image.C, image.H, image.W)
	reads := []THA3Tensor{image, grid}
	if add != nil {
		reads = append(reads, *add)
	}
	g.op(reads, []THA3Tensor{out}, func(r *Recorder, x *THA3Runner) {
		push := tha3WarpPush{image: x.at(image), grid: x.at(grid), dst: x.at(out),
			h: uint32(image.H), w: uint32(image.W), c: uint32(image.C)}
		if add != nil {
			push.add, push.hasAdd = x.at(*add), 1
		}
		x.dispatch(r, tha3Warp, groups256(image.H*image.W), 1, unsafe.Pointer(&push))
	})
	return out
}
