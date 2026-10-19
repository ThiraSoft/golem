package flux3

import (
	"fmt"

	"github.com/ThiraSoft/golem/krea2"
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
	Steps:     16,
	CFG:       1.0,
	Sampler:   "euler",
	Scheduler: "simple",
	Shift:     Shift, // ln(16), ModelSamplingFlux alpha 16.0
	Width:     1024,
	Height:    1024,
}

// StartingNoise returns the initial latent noise for a given seed and latent patch
// dimensions (h, w), 128 channels. MT19937 draws and layout match torch to the bit;
// the Box-Muller in float64 lands within a few ULP of torch's AVX2 float32 (76.65% exact,
// max diff 4.77e-7).
func StartingNoise(seed uint64, h, w int) []float32 {
	return krea2.TorchCPURandn(seed, 128*h*w)
}

// Sample runs the Euler ODE solver over sigmas, updating the latent in place.
// Euler adds no noise, so seed is accepted for interface compatibility with
// stochastic samplers (e.g. ER-SDE) and left unused.
// d is the denoiser returning the clean estimate for latent x at step and sigma.
func Sample(d krea2.Denoiser, x []float32, sigmas []float32, seed uint64, progress krea2.Progress) ([]float32, error) {
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

// ApplyCFG combines conditional and unconditional denoised estimates:
// uncond + (cond - uncond) * cfg.
// Returns an error if cond and uncond have different lengths.
func ApplyCFG(cond, uncond []float32, cfg float32) ([]float32, error) {
	if len(cond) != len(uncond) {
		return nil, fmt.Errorf("flux3: cond len %d != uncond len %d", len(cond), len(uncond))
	}
	if cfg == 1.0 {
		return cond, nil
	}
	out := make([]float32, len(cond))
	for i := range out {
		out[i] = uncond[i] + (cond[i]-uncond[i])*cfg
	}
	return out, nil
}
