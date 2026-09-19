package krea2

import (
	"math"
	"testing"
)

// What torch 2.11 printed on the ROCm machine, for seed 42.
func TestTorchCPURandnSeed42(t *testing.T) {
	got := TorchCPURandn(42, 16*32*32)
	want := []float32{1.92691529, 1.48728406, 0.900717199, -2.10552096, 0.678418458}
	near(t, "cpu", got[:len(want)], want, 1e-6)
}

func TestCUDARandnSeed42(t *testing.T) {
	g := NewCUDARandn(42)
	got := g.Next(16 * 64 * 64)
	want := []float32{0.19401902, 2.16137385, -0.172050655, 0.849060297, -1.92439914}
	near(t, "cuda", got[:len(want)], want, 2e-6)
}

func near(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}
