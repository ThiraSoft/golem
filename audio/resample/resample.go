// Package resample makes a signal mono and puts it at the rate a model wants.
//
// The kernel is a windowed sinc — a Blackman window over sixteen zero
// crossings a side — which is what every reference implementation of this
// resampling uses and is not expensive at these lengths: a second of audio is
// sixteen thousand outputs of sixty-four taps, microseconds on one core.
package resample

import "math"

const taps = 16 // zero crossings each side of an output's centre

// Mono averages the channels of an interleaved signal.
func Mono(interleaved []float32, channels int) []float32 {
	if channels <= 1 {
		out := make([]float32, len(interleaved))
		copy(out, interleaved)
		return out
	}
	out := make([]float32, len(interleaved)/channels)
	for i := range out {
		var sum float32
		for c := 0; c < channels; c++ {
			sum += interleaved[i*channels+c]
		}
		out[i] = sum / float32(channels)
	}
	return out
}

// To resamples, and copies when the rates are equal.
//
// Going down in rate stretches the sinc — and the window with it — by the
// ratio of the rates. That stretch is the low-pass filter: without it every
// frequency above the new Nyquist folds back into the signal as a tone that
// was never played, and nothing about the result looks wrong until someone
// listens to it.
func To(samples []float32, from, to int) []float32 {
	out := make([]float32, 0)
	if from <= 0 || to <= 0 {
		return out
	}
	if from == to {
		out = make([]float32, len(samples))
		copy(out, samples)
		return out
	}
	if len(samples) == 0 {
		return out
	}

	ratio := float64(to) / float64(from)
	cutoff := math.Min(1, ratio) // in cycles per input sample
	half := float64(taps) / cutoff

	n := int(math.Round(float64(len(samples)) * ratio))
	out = make([]float32, n)
	for i := range out {
		centre := float64(i) / ratio
		lo := int(math.Ceil(centre - half))
		hi := int(math.Floor(centre + half))
		if lo < 0 {
			lo = 0
		}
		if hi > len(samples)-1 {
			hi = len(samples) - 1
		}
		var sum, weight float64
		for j := lo; j <= hi; j++ {
			d := centre - float64(j)
			w := sinc(cutoff*d) * blackman(d/half)
			sum += float64(samples[j]) * w
			weight += w
		}
		if weight != 0 {
			out[i] = float32(sum / weight)
		}
	}
	return out
}

// sinc is the normalized cardinal sine, sin(pi x)/(pi x), which is 1 at zero.
func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

// blackman is the window over [-1, 1] that keeps the truncated sinc from
// ringing; outside that interval it is zero, which is what ends the sum.
func blackman(x float64) float64 {
	if x <= -1 || x >= 1 {
		return 0
	}
	t := math.Pi * (x + 1)
	return 0.42 - 0.5*math.Cos(t) + 0.08*math.Cos(2*t)
}
