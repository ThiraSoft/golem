package gemma

// The front of the conformer: two convolutions that quarter the time axis and
// the frequency axis together, then a projection that flattens what is left
// into the tower's width.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// audioPositions is what a number of mel frames becomes after the two strided
// convolutions: each is 3x3 with a stride of two and a padding of one, so each
// halves and rounds up.
func audioPositions(frames int) int {
	for i := 0; i < 2; i++ {
		frames = (frames-1)/2 + 1
	}
	return frames
}

// preEncode runs the subsampling over a mel of `frames` frames, laid out
// mel-major — value (m, t) at mel[m*frames+t] — and returns n positions of
// Cfg.Dim, the position the slower index.
//
// Both convolutions are 3x3 with a stride of two and a padding of one on every
// side: PyTorch's symmetric padding, not the causal padding the block's own
// convolution uses. What follows each is a LayerNorm over the channels alone —
// mean and variance across the channel axis at one point of the grid, which is
// not the RMS norm the rest of the tower uses — and a ReLU.
func (a *AudioTower) preEncode(melIn []float32, frames int) ([]float32, int) {
	// The convolution wants the grid the other way round from the front end:
	// [freq, time] with the frequency the faster index, where the mel comes
	// mel-major. llama.cpp opens its graph with exactly this transpose.
	freq, time := a.Cfg.MelBins, frames
	cur := make([]float32, freq*time)
	for m := 0; m < freq; m++ {
		for t := 0; t < time; t++ {
			cur[t*freq+m] = melIn[m*frames+t]
		}
	}
	for i := range a.W.Conv {
		c := a.W.Conv[i]
		outFreq, outTime := (freq-1)/2+1, (time-1)/2+1
		next := make([]float32, outFreq*outTime*c.Out)
		conv2dStride2Pad1(cur, next, c, freq, time, outFreq, outTime)
		layerNormOverChannels(next, c.Norm, outFreq*outTime, c.Out, a.Cfg.Eps)
		for j, v := range next {
			if v < 0 {
				next[j] = 0
			}
		}
		cur, freq, time = next, outFreq, outTime
	}

	// [freq, time, ch] flattened to one row of ch*freq per position, with the
	// channel the faster index inside the row and the frequency the slower —
	// which is the order clip.cpp's permute(1, 2, 0, 3) produces, and the
	// order the projection was fitted for. The other way round runs and gives
	// numbers that look plausible and are not.
	ch := a.W.Conv[1].Out
	flat := make([]float32, time*ch*freq)
	for t := 0; t < time; t++ {
		row := flat[t*ch*freq:]
		for c := 0; c < ch; c++ {
			plane := cur[(c*time+t)*freq:]
			for f := 0; f < freq; f++ {
				row[f*ch+c] = plane[f]
			}
		}
	}

	out := make([]float32, time*a.Cfg.Dim)
	a.W.InputProj.apply(flat, out, time)
	return out, time
}

// conv2dStride2Pad1 is the 3x3 convolution both subsampling layers are, with
// the input as [freq, time, inCh] and the output as [freq', time', outCh],
// frequency the faster index in each.
func conv2dStride2Pad1(in, out []float32, c AudioConv, freq, time, outFreq, outTime int) {
	nn.InParallel(c.Out, outFreq*outTime*c.Out*c.In*9, func(start, end int) {
		for oc := start; oc < end; oc++ {
			plane := out[oc*outFreq*outTime : (oc+1)*outFreq*outTime]
			for ot := 0; ot < outTime; ot++ {
				for of := 0; of < outFreq; of++ {
					var sum float32
					for ic := 0; ic < c.In; ic++ {
						// The kernel is stored [kw, kh, in, out], the first
						// dimension fastest, and kw runs along the frequency
						// axis because that is the grid's dimension 0.
						k := c.W[((oc*c.In+ic)*c.KH)*c.KW:]
						src := in[ic*freq*time:]
						for kh := 0; kh < c.KH; kh++ {
							tt := 2*ot + kh - 1
							if tt < 0 || tt >= time {
								continue
							}
							for kw := 0; kw < c.KW; kw++ {
								ff := 2*of + kw - 1
								if ff < 0 || ff >= freq {
									continue
								}
								sum += k[kh*c.KW+kw] * src[tt*freq+ff]
							}
						}
					}
					plane[ot*outFreq+of] = sum
				}
			}
		}
	})
}

// layerNormOverChannels normalizes each of the `points` positions of the grid
// across the `channels` planes, then scales by the gain.
//
// It is a mean-and-variance normalization, not an RMS one, and it runs across
// the channel axis while the data is laid out plane by plane. clip.cpp permutes
// the tensor twice to say the same thing.
func layerNormOverChannels(x, gain []float32, points, channels int, eps float32) {
	nn.InParallel(points, points*channels*2, func(start, end int) {
		for p := start; p < end; p++ {
			var mean float32
			for c := 0; c < channels; c++ {
				mean += x[c*points+p]
			}
			mean /= float32(channels)
			var variance float32
			for c := 0; c < channels; c++ {
				d := x[c*points+p] - mean
				variance += d * d
			}
			variance /= float32(channels)
			scale := float32(1 / math.Sqrt(float64(variance)+float64(eps)))
			for c := 0; c < channels; c++ {
				x[c*points+p] = (x[c*points+p] - mean) * scale * gain[c]
			}
		}
	})
}

// apply computes Y = clamp(W * clamp(X)) for a batch of rows.
func (l AudioLinear) apply(x, y []float32, batch int) {
	nn.InParallel(l.Outputs, batch*l.Inputs*l.Outputs, func(start, end int) {
		for b := 0; b < batch; b++ {
			row := x[b*l.Inputs : (b+1)*l.Inputs]
			dst := y[b*l.Outputs : (b+1)*l.Outputs]
			for o := start; o < end; o++ {
				w := l.W[o*l.Inputs : (o+1)*l.Inputs]
				var sum float32
				for i, v := range row {
					sum += w[i] * clampF32(v, l.Clamp.InMin, l.Clamp.InMax)
				}
				dst[o] = clampF32(sum, l.Clamp.OutMin, l.Clamp.OutMax)
			}
		}
	})
}
