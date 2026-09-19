package krea2

import "math"

// Shift is the flow shift Krea 2's model sampling is built with
// (comfy/supported_models.py, sampling_settings).
const Shift = 1.15

// flowTimesteps is how finely ModelSamplingFlux tabulates its sigmas.
const flowTimesteps = 10000

// flowSigma is flux_time_shift(shift, 1, t) as torch computes it on a float32
// tensor of t: exp(shift) is a Python float, the rest is float32.
func flowSigma(t float32) float32 {
	e := float32(math.Exp(Shift))
	return e / (e + (1/t - 1))
}

// Sigmas is ComfyUI's "simple" scheduler over ModelSamplingFlux: steps sigmas
// read back from the end of the table at even strides, then zero.
func Sigmas(steps int) []float32 {
	out := make([]float32, 0, steps+1)
	stride := float64(flowTimesteps) / float64(steps)
	for x := 0; x < steps; x++ {
		i := flowTimesteps - 1 - int(float64(x)*stride)
		out = append(out, flowSigma(float32(i+1)/flowTimesteps))
	}
	return append(out, 0)
}

// Latent statistics of the Wan 2.1 VAE (comfy/latent_formats.py, Wan21): the
// sampler works on (latent - mean) / std, and the decoder is handed the
// latent back.
var (
	latentMean = [16]float32{-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
		0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921}
	latentStd = [16]float32{2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
		3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160}
)

// LatentOut is process_latent_out on a 16 × plane latent, in place.
func LatentOut(latent []float32) {
	plane := len(latent) / 16
	for c := 0; c < 16; c++ {
		for i := c * plane; i < (c+1)*plane; i++ {
			latent[i] = latent[i]*latentStd[c] + latentMean[c]
		}
	}
}

func expShift() float64 { return 1 / math.Exp(Shift) }
