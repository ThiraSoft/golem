package flux3

import (
	"fmt"

	"github.com/ThiraSoft/golem/krea2"
)

// Defaults ruled from ref/flux3/workflows/flux3_klein_bf16_t2i.json.
const (
	DefaultSteps     = 16
	DefaultCFG       = 1.0
	DefaultSampler   = "euler"
	DefaultScheduler = "simple"
	DefaultShift     = Shift // ln(16), ModelSamplingFlux alpha 16.0
	DefaultWidth     = 1024
	DefaultHeight    = 1024
)

// Defaults holds the generation parameters from ref/flux3/workflows/flux3_klein_bf16_t2i.json.
var Defaults = struct {
	Steps     int
	CFG       float32
	Sampler   string
	Scheduler string
	Shift     float32
	Width     int
	Height    int
}{
	Steps:     DefaultSteps,
	CFG:       DefaultCFG,
	Sampler:   DefaultSampler,
	Scheduler: DefaultScheduler,
	Shift:     DefaultShift,
	Width:     DefaultWidth,
	Height:    DefaultHeight,
}

// StartingNoise returns the initial latent noise for a given seed and latent patch
// dimensions (h, w), 128 channels. It matches ComfyUI's CPU torch.randn bit for bit.
func StartingNoise(seed uint64, h, w int) []float32 {
	return krea2.TorchCPURandn(seed, 128*h*w)
}

// Sample runs the Euler ODE solver over sigmas, updating the latent in place.
// Euler adds no noise, so seed is preserved for interface compatibility.
// d is the denoiser returning the clean estimate for latent x at step and sigma.
func Sample(d krea2.Denoiser, x []float32, sigmas []float32, seed uint64, progress krea2.Progress) ([]float32, error) {
	_ = seed
	if len(sigmas) < 2 {
		return nil, fmt.Errorf("flux3: %d sigmas is no schedule", len(sigmas))
	}
	x = append([]float32(nil), x...)
	steps := len(sigmas) - 1
	for i := 0; i < steps; i++ {
		denoised, err := d(i, x, sigmas[i])
		if err != nil {
			return nil, err
		}
		if sigmas[i+1] == 0 {
			x = append([]float32(nil), denoised...)
		} else {
			s := sigmas[i]
			dt := sigmas[i+1] - s
			for k := range x {
				dVal := (x[k] - denoised[k]) / s
				x[k] += dVal * dt
			}
		}
		if progress != nil {
			progress(i+1, steps)
		}
	}
	return x, nil
}

// ApplyCFG combines conditional and unconditional denoised estimates.
func ApplyCFG(cond, uncond []float32, cfg float32) []float32 {
	if cfg == 1.0 || len(uncond) == 0 {
		return cond
	}
	out := make([]float32, len(cond))
	for i := range out {
		out[i] = uncond[i] + (cond[i]-uncond[i])*cfg
	}
	return out
}
