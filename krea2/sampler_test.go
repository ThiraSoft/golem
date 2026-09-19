package krea2

import (
	"fmt"
	"testing"
)

func TestSigmasMatchComfyUI(t *testing.T) {
	f := loadFixtures(t)
	compare(t, "sigmas", Sigmas(8), f.read(t, "sample/sigmas"), 1e-6)
}

func TestStartingNoiseMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	want := f.read(t, "sample/init")
	compare(t, "init", TorchCPURandn(7, len(want)), want, 1e-5)
}

func TestStepNoiseMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	g := NewCUDARandn(7)
	for i := 0; f.has(fmt.Sprintf("sample/noise%d", i)); i++ {
		want := f.read(t, fmt.Sprintf("sample/noise%d", i))
		compare(t, fmt.Sprintf("noise%d", i), g.Next(len(want)), want, 1e-5)
	}
}

// The sampler alone: the model's recorded answers are played back to it, and
// every latent it goes through must be the one ComfyUI went through.
func TestERSDEReplaysComfyUI(t *testing.T) {
	f := loadFixtures(t)
	sigmas := f.read(t, "sample/sigmas")
	steps := len(sigmas) - 1
	x0 := f.read(t, "sample/x0")
	var seen int
	d := func(step int, x []float32, sigma float32) ([]float32, error) {
		compare(t, fmt.Sprintf("x%d", step), x, f.read(t, fmt.Sprintf("sample/x%d", step)), 1e-4)
		seen++
		return f.read(t, fmt.Sprintf("sample/denoised%d", step)), nil
	}
	out, err := SampleERSDE(d, x0, sigmas, NewCUDARandn(7), nil)
	if err != nil {
		t.Fatal(err)
	}
	if seen != steps {
		t.Fatalf("%d model calls, want %d", seen, steps)
	}
	LatentOut(out)
	compare(t, "latent", out, f.read(t, "sample/latent"), 1e-4)
}
