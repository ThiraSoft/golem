package tha3

import "math"

// ReLU clamps negatives to zero, in place.
func ReLU(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

// LeakyReLU scales negatives by slope, in place.
func LeakyReLU(x []float32, slope float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = v * slope
		}
	}
}

// leaky is the rotator's and the editor's non-linearity: a slope of 0.1.
func leaky(x []float32) { LeakyReLU(x, 0.1) }

// Sigmoid is computed in float64 and rounded once, which is closer to
// PyTorch's float32 kernel than a float32 exponential would be.
func Sigmoid(x []float32) {
	for i, v := range x {
		x[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
}

// Tanh, in place, through float64 for the same reason as Sigmoid.
func Tanh(x []float32) {
	for i, v := range x {
		x[i] = float32(math.Tanh(float64(v)))
	}
}
