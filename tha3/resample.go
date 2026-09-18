package tha3

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// tap is where one output coordinate reads from: two neighbours and the
// weight of the second.
type tap struct {
	i0, i1 int
	f      float32
}

// bilinearTaps follows PyTorch's area_pixel_compute_source_index with
// align_corners=False: the output pixel centre mapped back, clamped at zero
// on the low side, and the upper neighbour clamped to the last pixel.
func bilinearTaps(in, out int) []tap {
	scale := float32(in) / float32(out)
	taps := make([]tap, out)
	for o := range taps {
		src := (float32(o)+0.5)*scale - 0.5
		if src < 0 {
			src = 0
		}
		i0 := min(int(src), in-1)
		i1 := min(i0+1, in-1)
		taps[o] = tap{i0, i1, src - float32(i0)}
	}
	return taps
}

// ResizeBilinear is F.interpolate(mode="bilinear", align_corners=False)
// without antialias, which is what the poser calls between its networks.
// nn.ResizeBilinearAntialias is a different filter and must not stand in.
func ResizeBilinear(x Tensor, h, w int) Tensor {
	out := NewTensor(x.C, h, w)
	ys, xs := bilinearTaps(x.H, h), bilinearTaps(x.W, w)
	nn.InParallel(x.C, x.C*h*w*4, func(start, end int) {
		for c := start; c < end; c++ {
			src, dst := x.Plane(c), out.Plane(c)
			for oy, ty := range ys {
				r0, r1 := src[ty.i0*x.W:], src[ty.i1*x.W:]
				for ox, tx := range xs {
					top := r0[tx.i0]*(1-tx.f) + r0[tx.i1]*tx.f
					bottom := r1[tx.i0]*(1-tx.f) + r1[tx.i1]*tx.f
					dst[oy*w+ox] = top*(1-ty.f) + bottom*ty.f
				}
			}
		}
	})
	return out
}

// UpsampleNearest2 doubles both sides, each pixel becoming a 2×2 block:
// Upsample(scale_factor=2, mode="nearest").
func UpsampleNearest2(x Tensor) Tensor {
	out := NewTensor(x.C, 2*x.H, 2*x.W)
	w := 2 * x.W
	for c := 0; c < x.C; c++ {
		src, dst := x.Plane(c), out.Plane(c)
		for y := 0; y < 2*x.H; y++ {
			row := src[(y/2)*x.W:]
			for ox := 0; ox < w; ox++ {
				dst[y*w+ox] = row[ox/2]
			}
		}
	}
	return out
}

// applyGridChange warps image by an offset field, as GridChangeApplier does:
// the identity grid of affine_grid(align_corners=False), plus change, sampled
// by grid_sample(mode="bilinear", padding_mode="border", align_corners=False).
//
// change holds two planes, the x offset then the y offset, in normalized
// units where the image spans [-1, 1]. The identity grid puts pixel j at
// (2j+1)/W - 1, and a normalized coordinate u maps back to ((u+1)W - 1)/2,
// which the border mode clamps to [0, W-1] before interpolating.
func applyGridChange(change, image Tensor) Tensor {
	h, w := image.H, image.W
	out := NewTensor(image.C, h, w)
	gx, gy := change.Plane(0), change.Plane(1)
	nn.InParallel(h, image.C*h*w*8, func(start, end int) {
		for y := start; y < end; y++ {
			for x := 0; x < w; x++ {
				p := y*w + x
				u := (2*float32(x)+1)/float32(w) - 1 + gx[p]
				v := (2*float32(y)+1)/float32(h) - 1 + gy[p]
				sx := clamp(((u+1)*float32(w)-1)/2, 0, float32(w-1))
				sy := clamp(((v+1)*float32(h)-1)/2, 0, float32(h-1))
				x0, y0 := int(math.Floor(float64(sx))), int(math.Floor(float64(sy)))
				fx, fy := sx-float32(x0), sy-float32(y0)
				x1, y1 := x0+1, y0+1
				for c := 0; c < image.C; c++ {
					plane := image.Plane(c)
					at := func(xx, yy int) float32 {
						if xx >= w || yy >= h {
							return 0
						}
						return plane[yy*w+xx]
					}
					out.Plane(c)[p] = at(x0, y0)*(1-fx)*(1-fy) + at(x1, y0)*fx*(1-fy) +
						at(x0, y1)*(1-fx)*fy + at(x1, y1)*fx*fy
				}
			}
		}
	})
	return out
}

func clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
