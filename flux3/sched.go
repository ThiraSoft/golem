package flux3

import "math"

// Shift is the flow shift from ref/flux3/workflows/flux3_klein_bf16_t2i.json:
// ModelSamplingFlux with alpha 16.0 gives shift ln(16) = 2.772588722239781.
// Note that supported_models.py defines a default shift of 1.93586, but the
// upstream BFL workflow overrides it with alpha 16.
const Shift = 2.772588722239781

// flowTimesteps is how finely ModelSamplingFlux tabulates its sigmas in ComfyUI.
const flowTimesteps = 10000

// flowSigma is flux_time_shift(shift, 1, t) as torch computes it on a float32 tensor:
// exp(shift) is a Python float, the rest is float32.
func flowSigma(t float32) float32 {
	e := float32(math.Exp(Shift))
	return e / (e + (1/t - 1))
}

// Sigmas is ComfyUI's "simple" scheduler over ModelSamplingFlux: steps sigmas
// read back from the end of the table at even strides, then zero.
func Sigmas(steps int) []float32 {
	if steps <= 0 {
		return []float32{0}
	}
	out := make([]float32, 0, steps+1)
	stride := float64(flowTimesteps) / float64(steps)
	for x := 0; x < steps; x++ {
		i := flowTimesteps - 1 - int(float64(x)*stride)
		out = append(out, flowSigma(float32(i+1)/flowTimesteps))
	}
	return append(out, 0)
}

// NextSigma returns the schedule's next sigma for step (or the final sigma at the last step).
func NextSigma(sigmas []float32, step int) float32 {
	if step+1 < len(sigmas) {
		return sigmas[step+1]
	}
	if len(sigmas) > 0 {
		return sigmas[len(sigmas)-1]
	}
	return 0
}
