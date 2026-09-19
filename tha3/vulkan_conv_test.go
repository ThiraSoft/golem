package tha3

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/vk"
)

func TestCardDepthwise(t *testing.T) {
	r := rand.New(rand.NewSource(20))
	for _, geo := range []struct{ k, stride, pad int }{{3, 1, 1}, {4, 2, 1}, {7, 1, 3}} {
		x := randomTensor(r, 5, 17, 19)
		wt := random(r, 5*geo.k*geo.k)
		c := Conv2d{In: 5, Out: 5, K: geo.k, Stride: geo.stride, Pad: geo.pad, Groups: 5, Weight: wt}
		got := onCard(t, nil, []Tensor{x}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
			return g.DW(in[0], g.Weights(wt), geo.k, geo.stride, geo.pad)
		})
		checkCard(t, fmt.Sprintf("depthwise k%d s%d", geo.k, geo.stride), got, c.Apply(x))
	}
}

func TestCardTransposed(t *testing.T) {
	r := rand.New(rand.NewSource(21))
	x := randomTensor(r, 6, 9, 7)
	wt := random(r, 6*16)
	c := ConvTranspose2d{C: 6, K: 4, Stride: 2, Pad: 1, Weight: wt}
	got := onCard(t, nil, []Tensor{x}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
		return g.DWT(in[0], g.Weights(wt))
	})
	checkCard(t, "transposed", got, c.Apply(x))
}

// Sizes that are not multiples of the 64-wide tiles, so the edges of the
// product are exercised on both axes.
func TestCardPointwise(t *testing.T) {
	r := rand.New(rand.NewSource(22))
	x := randomTensor(r, 70, 33, 31)
	wt, bias := random(r, 130*70), random(r, 130)
	for _, withBias := range []bool{true, false} {
		c := Conv2d{In: 70, Out: 130, K: 1, Stride: 1, Pad: 0, Groups: 1, Weight: wt}
		if withBias {
			c.Bias = bias
		}
		got := onCard(t, nil, []Tensor{x}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
			b := vk.THA3Weights{}
			if withBias {
				b = g.Weights(bias)
			}
			return g.PW(in[0], g.Weights(wt), b, 130)
		})
		checkCard(t, fmt.Sprintf("pointwise bias=%v", withBias), got, c.Apply(x))
	}
}

func TestCardHeads(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	x := randomTensor(r, 64, 21, 23)
	for _, h := range []struct {
		out  int
		act  vk.THA3Act
		cpu  func([]float32)
		bias bool
	}{{4, vk.THA3Tanh, Tanh, true}, {1, vk.THA3Sigmoid, Sigmoid, true}, {2, vk.THA3None, nil, false}} {
		wt, bias := random(r, h.out*64*9), random(r, h.out)
		c := Conv2d{In: 64, Out: h.out, K: 3, Stride: 1, Pad: 1, Groups: 1, Weight: wt}
		if h.bias {
			c.Bias = bias
		}
		want := c.Apply(x)
		if h.cpu != nil {
			h.cpu(want.Data)
		}
		got := onCard(t, nil, []Tensor{x}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
			b := vk.THA3Weights{}
			if h.bias {
				b = g.Weights(bias)
			}
			return g.Head(in[0], g.Weights(wt), b, h.out, h.act)
		})
		checkCard(t, fmt.Sprintf("head %d", h.out), got, want)
	}
}

// The first case is wider than one statistics slice (4096 pixels), so the
// partial statistics are combined; the others fit one.
func TestCardNorm(t *testing.T) {
	r := rand.New(rand.NewSource(24))
	for _, n := range []struct {
		c, h, w  int
		act      vk.THA3Act
		residual bool
	}{{3, 300, 300, vk.THA3ReLU, true}, {5, 16, 16, vk.THA3Leaky, false}, {2, 64, 64, vk.THA3None, false}} {
		x, res := randomTensor(r, n.c, n.h, n.w), randomTensor(r, n.c, n.h, n.w)
		for i := range x.Data {
			x.Data[i] = x.Data[i]*3 + 0.5 // a mean and a spread worth normalizing
		}
		gamma, beta := random(r, n.c), random(r, n.c)
		want := x.Clone()
		InstanceNorm{Weight: gamma, Bias: beta}.Apply(want)
		switch n.act {
		case vk.THA3ReLU:
			ReLU(want.Data)
		case vk.THA3Leaky:
			leaky(want.Data)
		}
		if n.residual {
			for i, v := range res.Data {
				want.Data[i] += v
			}
		}
		got := onCard(t, nil, []Tensor{x, res}, func(g *vk.THA3Graph, in []vk.THA3Tensor) vk.THA3Tensor {
			var residual *vk.THA3Tensor
			if n.residual {
				residual = &in[1]
			}
			return g.Norm(in[0], g.Weights(gamma), g.Weights(beta), n.act, residual)
		})
		checkCard(t, fmt.Sprintf("norm %dx%dx%d", n.c, n.h, n.w), got, want)
	}
}
