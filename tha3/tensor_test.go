package tha3

import (
	"math"
	"testing"
)

func seq(c, h, w int) Tensor {
	t := NewTensor(c, h, w)
	for i := range t.Data {
		t.Data[i] = float32(i)
	}
	return t
}

func TestCropPaste(t *testing.T) {
	src := seq(2, 4, 4)
	crop := src.Crop(1, 2, 2, 2)
	want := []float32{6, 7, 10, 11, 22, 23, 26, 27}
	for i, v := range want {
		if crop.Data[i] != v {
			t.Fatalf("crop[%d] = %v, want %v", i, crop.Data[i], v)
		}
	}
	dst := NewTensor(2, 4, 4)
	dst.Paste(crop, 1, 2)
	if got := dst.Crop(1, 2, 2, 2); got.Data[3] != 11 || got.Data[7] != 27 || dst.Data[0] != 0 {
		t.Fatalf("paste did not put the crop back where it came from: %v", dst.Data)
	}
}

func TestConcatBroadcast(t *testing.T) {
	a := seq(1, 2, 2)
	b := Broadcast([]float32{7, 8}, 2, 2)
	c := Concat(a, b)
	want := []float32{0, 1, 2, 3, 7, 7, 7, 7, 8, 8, 8, 8}
	if c.C != 3 {
		t.Fatalf("C = %d", c.C)
	}
	for i, v := range want {
		if c.Data[i] != v {
			t.Fatalf("concat[%d] = %v, want %v", i, c.Data[i], v)
		}
	}
	if v := c.Channels(1, 3); v.C != 2 || v.Data[0] != 7 {
		t.Fatalf("Channels view wrong: %+v", v)
	}
}

func TestActivations(t *testing.T) {
	x := []float32{-2, 0, 3}
	LeakyReLU(x, 0.1)
	if x[0] != -0.2 || x[2] != 3 {
		t.Fatalf("leaky: %v", x)
	}
	ReLU(x)
	if x[0] != 0 {
		t.Fatalf("relu: %v", x)
	}
	y := []float32{0}
	Sigmoid(y)
	if y[0] != 0.5 {
		t.Fatalf("sigmoid(0) = %v", y[0])
	}
	z := []float32{1}
	Tanh(z)
	if math.Abs(float64(z[0])-math.Tanh(1)) > 1e-7 {
		t.Fatalf("tanh(1) = %v", z[0])
	}
}

func TestBlend(t *testing.T) {
	image := Broadcast([]float32{0, 0, 0, -1}, 1, 1)
	change := Broadcast([]float32{1, 1, 1, 1}, 1, 1)
	alpha := Broadcast([]float32{0.25}, 1, 1)
	got := applyColorChange(alpha, change, image)
	for i, v := range []float32{0.25, 0.25, 0.25, -0.5} {
		if got.Data[i] != v {
			t.Fatalf("color change[%d] = %v, want %v", i, got.Data[i], v)
		}
	}
	rgb := applyRGBChange(alpha, change, image)
	for i, v := range []float32{0.25, 0.25, 0.25, -1} {
		if rgb.Data[i] != v {
			t.Fatalf("rgb change[%d] = %v, want %v", i, rgb.Data[i], v)
		}
	}
	alpha4 := Broadcast([]float32{1, 0, 1, 0}, 1, 1)
	per := applyColorChange(alpha4, change, image)
	for i, v := range []float32{1, 0, 1, -1} {
		if per.Data[i] != v {
			t.Fatalf("per-channel alpha[%d] = %v, want %v", i, per.Data[i], v)
		}
	}
}
