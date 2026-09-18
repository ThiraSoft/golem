package tha3

import (
	"math"
	"math/rand"
	"testing"
)

func random(r *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = r.Float32()*2 - 1
	}
	return out
}

func randomTensor(r *rand.Rand, c, h, w int) Tensor {
	return Tensor{C: c, H: h, W: w, Data: random(r, c*h*w)}
}

func closeTo(t *testing.T, name string, got, want []float32, eps float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > eps {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

// A 3×3 box filter over a ramp, with padding 1: the corner sees four pixels.
func TestConvDenseByHand(t *testing.T) {
	x := seq(1, 3, 3)
	w := make([]float32, 9)
	for i := range w {
		w[i] = 1
	}
	c := Conv2d{In: 1, Out: 1, K: 3, Stride: 1, Pad: 1, Groups: 1, Weight: w, Bias: []float32{0.5}}
	got := c.Apply(x)
	want := []float32{8.5, 15.5, 12.5, 21.5, 36.5, 27.5, 20.5, 33.5, 24.5}
	closeTo(t, "box", got.Data, want, 1e-6)
}

// A depthwise convolution is a dense one whose kernel is zero off the diagonal.
func TestConvDepthwiseMatchesDense(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, geometry := range []struct{ k, stride, pad int }{{3, 1, 1}, {4, 2, 1}, {7, 1, 3}} {
		const ch = 5
		x := randomTensor(r, ch, 12, 12)
		dw := Conv2d{In: ch, Out: ch, K: geometry.k, Stride: geometry.stride, Pad: geometry.pad, Groups: ch,
			Weight: random(r, ch*geometry.k*geometry.k)}
		kk := geometry.k * geometry.k
		dense := Conv2d{In: ch, Out: ch, K: geometry.k, Stride: geometry.stride, Pad: geometry.pad, Groups: 1,
			Weight: make([]float32, ch*ch*kk)}
		for c := 0; c < ch; c++ {
			copy(dense.Weight[(c*ch+c)*kk:(c*ch+c+1)*kk], dw.Weight[c*kk:(c+1)*kk])
		}
		closeTo(t, "depthwise", dw.Apply(x).Data, dense.Apply(x).Data, 1e-5)
	}
}

// A 1×1 convolution is a matrix product over the channels.
func TestConvPointwiseMatchesDense(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	x := randomTensor(r, 6, 5, 7)
	w := random(r, 3*6)
	b := random(r, 3)
	pointwise := Conv2d{In: 6, Out: 3, K: 1, Stride: 1, Pad: 0, Groups: 1, Weight: w, Bias: b}
	got := pointwise.Apply(x)
	for o := 0; o < 3; o++ {
		for p := 0; p < 35; p++ {
			want := b[o]
			for i := 0; i < 6; i++ {
				want += w[o*6+i] * x.Data[i*35+p]
			}
			if math.Abs(float64(got.Data[o*35+p]-want)) > 1e-5 {
				t.Fatalf("out[%d][%d] = %v, want %v", o, p, got.Data[o*35+p], want)
			}
		}
	}
}

// The transposed convolution is the adjoint of the convolution with the same
// kernel: <convT(x), y> equals <x, conv(y)> for every x and y.
func TestConvTransposeIsAdjoint(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	const ch = 3
	w := random(r, ch*16)
	up := ConvTranspose2d{C: ch, K: 4, Stride: 2, Pad: 1, Weight: w}
	down := Conv2d{In: ch, Out: ch, K: 4, Stride: 2, Pad: 1, Groups: ch, Weight: w}
	x := randomTensor(r, ch, 6, 6)
	y := randomTensor(r, ch, 12, 12)
	ux := up.Apply(x)
	if ux.H != 12 || ux.W != 12 {
		t.Fatalf("transposed output %dx%d, want 12x12", ux.H, ux.W)
	}
	dy := down.Apply(y)
	var left, right float64
	for i := range ux.Data {
		left += float64(ux.Data[i]) * float64(y.Data[i])
	}
	for i := range x.Data {
		right += float64(x.Data[i]) * float64(dy.Data[i])
	}
	if math.Abs(left-right) > 1e-3 {
		t.Fatalf("<convT(x), y> = %v, <x, conv(y)> = %v", left, right)
	}
}
