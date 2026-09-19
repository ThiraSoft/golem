package krea2

import (
	"fmt"
	"math"
)

// Denoiser is the model as a sampler sees it: the latent at a noise level in,
// the model's estimate of the clean latent out. For a flow model that is
// x - sigma·v, v being what the DiT predicts.
type Denoiser func(step int, x []float32, sigma float32) ([]float32, error)

// Progress, if set, hears about each step as the sampler finishes it.
type Progress func(step, steps int)

// SampleERSDE is ComfyUI's sample_er_sde (comfy/k_diffusion/sampling.py): the
// extended reverse-time SDE solver of Cui et al., three stages, its default
// noise scaler, and the noise drawn from noise at every step but the last.
// x is the starting latent (pure noise at the first sigma) and is not kept.
func SampleERSDE(d Denoiser, x []float32, sigmas []float32, noise *CUDARandn, progress Progress) ([]float32, error) {
	if len(sigmas) < 2 {
		return nil, fmt.Errorf("krea2: %d sigmas is no schedule", len(sigmas))
	}
	sigmas = append([]float32(nil), sigmas...)
	// offset_first_sigma_for_snr: a flow model's first sigma of 1 has no
	// log-SNR, so it is moved to percent_to_sigma(1e-4).
	if sigmas[0] >= 1 {
		sigmas[0] = float32(math.Exp(Shift) / (math.Exp(Shift) + (1/(1-1e-4) - 1)))
	}
	// er_lambda = sigma / alpha, alpha = 1 - sigma for a flow model.
	lambdas := make([]float32, len(sigmas))
	for i, s := range sigmas {
		lambdas[i] = float32(math.Exp(-float64(logitNeg(s))))
	}

	const points = 200
	x = append([]float32(nil), x...)
	var oldDenoised, oldD []float32
	steps := len(sigmas) - 1
	for i := 0; i < steps; i++ {
		denoised, err := d(i, x, sigmas[i])
		if err != nil {
			return nil, err
		}
		stage := min(3, i+1)
		if sigmas[i+1] == 0 {
			x = denoised
		} else {
			ls, lt := lambdas[i], lambdas[i+1]
			alphaS := sigmas[i] / ls
			alphaT := sigmas[i+1] / lt
			rAlpha := alphaT / alphaS
			r := noiseScaler(lt) / noiseScaler(ls)
			for k := range x {
				x[k] = rAlpha*r*x[k] + alphaT*(1-r)*denoised[k]
			}
			var dnow []float32
			if stage >= 2 {
				dt := lt - ls
				step := -dt / points
				var s, su float32
				for p := 0; p < points; p++ {
					pos := lt + float32(p)*step
					scaled := noiseScaler(pos)
					s += 1 / scaled
					su += (pos - ls) / scaled
				}
				s *= step
				su *= step
				dnow = make([]float32, len(x))
				div := ls - lambdas[i-1]
				c2 := alphaT * (dt + s*noiseScaler(lt))
				for k := range x {
					dnow[k] = (denoised[k] - oldDenoised[k]) / div
					x[k] += c2 * dnow[k]
				}
				if stage >= 3 {
					div3 := (ls - lambdas[i-2]) / 2
					c3 := alphaT * (dt*dt/2 + su*noiseScaler(lt))
					for k := range x {
						x[k] += c3 * (dnow[k] - oldD[k]) / div3
					}
				}
				oldD = dnow
			}
			amp := lt*lt - ls*ls*r*r
			scale := float32(0)
			if amp > 0 {
				scale = alphaT * float32(math.Sqrt(float64(amp)))
			}
			n := noise.Next(len(x))
			for k := range x {
				x[k] += n[k] * scale
			}
		}
		oldDenoised = denoised
		if progress != nil {
			progress(i+1, steps)
		}
	}
	return x, nil
}

// logitNeg is sigma_to_half_log_snr for a flow model: -logit(sigma).
func logitNeg(s float32) float32 {
	return float32(-math.Log(float64(s) / (1 - float64(s))))
}

func noiseScaler(x float32) float32 {
	return x * (float32(math.Exp(math.Pow(float64(x), 0.3))) + 10)
}
