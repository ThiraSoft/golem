package nn

// Elementary transformer operations, written over plain vectors: at batch 1 and
// one timestep at a time, there is never a tensor of rank higher than two to
// handle, and the indices stay readable.

import "math"

// LayerNorm normalizes x in place over its last (and only) dimension, then
// applies gain and bias. eps is 1e-5 throughout the model.
type LayerNorm struct {
	Gain, Bias []float32
	Eps        float32
}

func (n LayerNorm) Apply(x []float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := sum / float64(len(x))

	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(len(x))

	inv := 1 / math.Sqrt(variance+float64(n.Eps))
	for i, v := range x {
		x[i] = float32((float64(v)-mean)*inv)*n.Gain[i] + n.Bias[i]
	}
}

// GELU applies the exact variant (based on the error function), the one
// `torch.nn.functional.gelu` uses by default. The hyperbolic-tangent
// approximation would drift from the reference by the third decimal.
func GELU(x []float32) {
	// The error function is expensive: four thousand calls per layer,
	// twenty-four layers per frame. Spreading over the cores is justified here,
	// where it would not be for an addition.
	InParallel(len(x), len(x)*128, func(start, end int) {
		GELURange(x, start, end)
	})
}

// GELURange is the same over one stretch, on the caller's thread: the section
// that computed those values applies it before its barrier.
func GELURange(x []float32, start, end int) {
	for i := start; i < end; i++ {
		d := float64(x[i])
		x[i] = float32(0.5 * d * (1 + math.Erf(d/math.Sqrt2)))
	}
}

// SoftmaxInPlace normalizes x into probabilities, subtracting the maximum so
// that the exponential cannot overflow.
func SoftmaxInPlace(x []float32) {
	max := float32(math.Inf(-1))
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - max)))
		x[i] = e
		sum += e
	}
	for i := range x {
		x[i] /= sum
	}
}

// ApplyRoPE rotates the vector of one head (even dimension, treated as
// consecutive complex numbers) by an angle proportional to its absolute
// position in the sequence.
func ApplyRoPE(vec []float32, position int, maxPeriod float64) {
	d := len(vec)
	for i := 0; i < d/2; i++ {
		freq := math.Exp(float64(i) * (-math.Log(maxPeriod) * 2 / float64(d)))
		angle := freq * float64(position)
		cos, sin := math.Cos(angle), math.Sin(angle)
		re, im := float64(vec[2*i]), float64(vec[2*i+1])
		vec[2*i] = float32(re*cos - im*sin)
		vec[2*i+1] = float32(re*sin + im*cos)
	}
}

// SiLU, or "swish": x*sigma(x). This is the activation of the flow net and of
// the adaptive modulations, where pockettts's transformer uses GELU.
//
// The exponential is float64 and exact. A language engine reading a GGUF wants
// the other one — ggml's polynomial, in SwiGLURange — because there the
// reference is ggml and the two differ by a couple of units in the last place.
// Here the reference is PyTorch, and this is left alone.
func SiLU(x []float32) {
	for i, v := range x {
		x[i] = float32(float64(v) / (1 + math.Exp(-float64(v))))
	}
}

// ELU is the activation of the SEANet decoder, with alpha = 1.
//
// It is spread over the cores like a product would be, which is unusual for an
// activation and is here the right thing: the decoder applies it to a hundred
// and twenty thousand samples per frame, an exponential each for the negative
// half, and left on one thread that was close to a quarter of the frame while
// seven cores waited for it. An exponential is worth some sixty multiply-adds,
// which is what elementwiseWork counts it as when deciding whether the split
// pays.
func ELU(x []float32) {
	InParallel(len(x), len(x)*elementwiseWork, func(start, end int) {
		ELURange(x, start, end)
	})
}

// ELURange is the same over one stretch, on the caller's thread.
func ELURange(x []float32, start, end int) {
	for i, v := range x[start:end] {
		if v < 0 {
			x[start+i] = float32(math.Exp(float64(v)) - 1)
		}
	}
}

// elementwiseWork is what one transcendental costs, in units of the
// multiply-add that parallelThreshold is expressed in.
const elementwiseWork = 64

// RMSNorm divides by the standard deviation without subtracting the mean, then
// applies a gain. Mind the detail that matters: the variance is PyTorch's
// default, so *unbiased* — divided by n-1, not by n as in LayerNorm. A factor
// that would go unnoticed by eye and throw everything off.
type RMSNorm struct {
	Alpha []float32
	Eps   float32
}

func (n RMSNorm) Apply(x []float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := sum / float64(len(x))
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(len(x) - 1)

	inv := 1 / math.Sqrt(variance+float64(n.Eps))
	for i, v := range x {
		x[i] = float32(float64(v)*inv) * n.Alpha[i]
	}
}

// NormalizeNoAffine centers and scales without gain or bias — the final
// normalization of the flow net, whose modulation stands in for parameters.
func NormalizeNoAffine(x []float32, eps float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v)
	}
	mean := sum / float64(len(x))
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(len(x))
	inv := 1 / math.Sqrt(variance+float64(eps))
	for i, v := range x {
		x[i] = float32((float64(v) - mean) * inv)
	}
}

// RMSNormPlain is the normalization ggml performs: divide by the root mean
// square, with no mean subtracted and no correction to the denominator, then
// apply a gain if there is one.
//
// This is not what RMSNorm above does. That one centres the vector and divides
// by an unbiased variance, because that is what Moshi's PyTorch does, and the
// two disagree on every input whose mean is not zero. Both are correct; they
// belong to different models.
// The arithmetic follows ggml's to the letter: the squares are float32 products
// accumulated in double, the mean and the reciprocal square root are float32,
// and the scaling and the gain are applied in float32, in that order. Carrying
// the double through the multiplications instead moves the result by 1e-5,
// which is small until it decides which side of a rounding an attention score
// falls on.
func RMSNormPlain(x []float32, gain []float32, eps float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v * v)
	}
	mean := float32(sum / float64(len(x)))
	scale := float32(1 / math.Sqrt(float64(mean+eps)))
	if gain == nil {
		for i, v := range x {
			x[i] = v * scale
		}
		return
	}
	if len(gain) != len(x) {
		panic("nn: RMSNormPlain gain length mismatch")
	}
	for i, v := range x {
		x[i] = v * scale * gain[i]
	}
}

// Softcap squashes values into (-limit, limit) smoothly. Gemma applies it to
// the logits, so that no token can run away.
func Softcap(x []float32, limit float32) {
	if limit == 0 {
		return
	}
	inv := 1 / float64(limit)
	for i, v := range x {
		x[i] = float32(float64(limit) * math.Tanh(float64(v)*inv))
	}
}
