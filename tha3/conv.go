package tha3

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
)

// Conv2d is PyTorch's Conv2d over a batch of one. Weight is laid out
// [Out][In/Groups][K][K] as in the state dict, and Bias is nil when the layer
// has none.
//
// These networks use two groupings and no other: 1, and In with Out == In
// (depthwise). Each gets its own loop, and a 1×1 dense layer a third, because
// the three have nothing in common worth sharing once written for speed.
type Conv2d struct {
	In, Out, K, Stride, Pad, Groups int
	Weight, Bias                    []float32
}

func (c Conv2d) outSize(n int) int { return (n+2*c.Pad-c.K)/c.Stride + 1 }

// Apply returns the convolution of x.
func (c Conv2d) Apply(x Tensor) Tensor {
	if x.C != c.In {
		panic(fmt.Sprintf("tha3: convolution of %d channels given %d", c.In, x.C))
	}
	switch {
	case c.Groups == c.In && c.Out == c.In && c.In > 1:
		return c.depthwise(x)
	case c.Groups == 1 && c.K == 1 && c.Stride == 1 && c.Pad == 0:
		return c.pointwise(x)
	case c.Groups == 1:
		return c.dense(x)
	}
	panic(fmt.Sprintf("tha3: convolution with %d groups over %d channels", c.Groups, c.In))
}

// pointwise is a product of the [Out][In] weight by the [In][H*W] picture,
// one output plane per task.
func (c Conv2d) pointwise(x Tensor) Tensor {
	out := NewTensor(c.Out, x.H, x.W)
	n := x.H * x.W
	nn.InParallel(c.Out, c.Out*c.In*n, func(start, end int) {
		for o := start; o < end; o++ {
			dst := out.Plane(o)
			if c.Bias != nil {
				for i := range dst {
					dst[i] = c.Bias[o]
				}
			}
			for i, weight := range c.Weight[o*c.In : (o+1)*c.In] {
				for p, v := range x.Plane(i) {
					dst[p] += weight * v
				}
			}
		}
	})
	return out
}

// depthwise convolves each channel with its own kernel, one channel per task.
func (c Conv2d) depthwise(x Tensor) Tensor {
	oh, ow := c.outSize(x.H), c.outSize(x.W)
	out := NewTensor(c.Out, oh, ow)
	k := c.K
	nn.InParallel(c.In, c.In*oh*ow*k*k, func(start, end int) {
		for ch := start; ch < end; ch++ {
			src, dst := x.Plane(ch), out.Plane(ch)
			w := c.Weight[ch*k*k : (ch+1)*k*k]
			var bias float32
			if c.Bias != nil {
				bias = c.Bias[ch]
			}
			for oy := 0; oy < oh; oy++ {
				for ox := 0; ox < ow; ox++ {
					sum := bias
					for ky := 0; ky < k; ky++ {
						iy := oy*c.Stride - c.Pad + ky
						if iy < 0 || iy >= x.H {
							continue
						}
						row := src[iy*x.W:]
						for kx := 0; kx < k; kx++ {
							ix := ox*c.Stride - c.Pad + kx
							if ix < 0 || ix >= x.W {
								continue
							}
							sum += w[ky*k+kx] * row[ix]
						}
					}
					dst[oy*ow+ox] = sum
				}
			}
		}
	})
	return out
}

// dense is the general case, which only the output heads reach: a 3×3 from a
// wide feature map to one, two or four channels. There are too few output
// channels to split on, so the work is split by output rows.
func (c Conv2d) dense(x Tensor) Tensor {
	oh, ow := c.outSize(x.H), c.outSize(x.W)
	out := NewTensor(c.Out, oh, ow)
	k := c.K
	nn.InParallel(oh, c.Out*c.In*oh*ow*k*k, func(start, end int) {
		for o := 0; o < c.Out; o++ {
			dst := out.Plane(o)
			if c.Bias != nil {
				for i := start * ow; i < end*ow; i++ {
					dst[i] = c.Bias[o]
				}
			}
			for i := 0; i < c.In; i++ {
				src := x.Plane(i)
				w := c.Weight[(o*c.In+i)*k*k : (o*c.In+i+1)*k*k]
				for ky := 0; ky < k; ky++ {
					for kx := 0; kx < k; kx++ {
						weight := w[ky*k+kx]
						for oy := start; oy < end; oy++ {
							iy := oy*c.Stride - c.Pad + ky
							if iy < 0 || iy >= x.H {
								continue
							}
							row, drow := src[iy*x.W:], dst[oy*ow:]
							for ox := 0; ox < ow; ox++ {
								ix := ox*c.Stride - c.Pad + kx
								if ix < 0 || ix >= x.W {
									continue
								}
								drow[ox] += weight * row[ix]
							}
						}
					}
				}
			}
		}
	})
	return out
}

// ConvTranspose2d is PyTorch's ConvTranspose2d restricted to what the
// upsampling blocks use: depthwise, no bias, Weight laid out [C][1][K][K].
// Every input pixel scatters its kernel into the output at stride Stride,
// shifted back by Pad.
type ConvTranspose2d struct {
	C, K, Stride, Pad int
	Weight            []float32
}

// Apply returns the transposed convolution of x.
func (c ConvTranspose2d) Apply(x Tensor) Tensor {
	if x.C != c.C {
		panic(fmt.Sprintf("tha3: transposed convolution of %d channels given %d", c.C, x.C))
	}
	oh := (x.H-1)*c.Stride - 2*c.Pad + c.K
	ow := (x.W-1)*c.Stride - 2*c.Pad + c.K
	out := NewTensor(c.C, oh, ow)
	k := c.K
	nn.InParallel(c.C, c.C*x.H*x.W*k*k, func(start, end int) {
		for ch := start; ch < end; ch++ {
			src, dst := x.Plane(ch), out.Plane(ch)
			w := c.Weight[ch*k*k : (ch+1)*k*k]
			for iy := 0; iy < x.H; iy++ {
				for ix := 0; ix < x.W; ix++ {
					v := src[iy*x.W+ix]
					for ky := 0; ky < k; ky++ {
						oy := iy*c.Stride - c.Pad + ky
						if oy < 0 || oy >= oh {
							continue
						}
						for kx := 0; kx < k; kx++ {
							ox := ix*c.Stride - c.Pad + kx
							if ox < 0 || ox >= ow {
								continue
							}
							dst[oy*ow+ox] += v * w[ky*k+kx]
						}
					}
				}
			}
		}
	})
	return out
}
