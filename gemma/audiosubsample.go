package gemma

// The front of the conformer: two convolutions that quarter the time axis and
// the frequency axis together, then a projection that flattens what is left
// into the tower's width.
//
// Both convolutions are 3x3 with a stride of two and a padding of one on every
// side: PyTorch's symmetric padding, not the causal padding the block's own
// convolution uses. What follows each is a LayerNorm over the channels alone —
// mean and variance across the channel axis at one point of the grid, which is
// not the RMS norm the rest of the tower uses — and a ReLU.
//
// The grid is held **point-major**: the channels of one point of the grid sit
// next to each other, and the points run frequency-fastest. Written the other
// way round — a plane per channel, which is how the file stores the weights and
// how the graph's tensors are shaped — every one of the three steps here reads
// with a stride. Point-major, the im2col gather is nine contiguous runs, the
// LayerNorm is a contiguous pass over what a dot product just produced, and the
// flatten at the end is a copy. That choice is most of why this file stopped
// being a quarter of the encoder's time.

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
func (a *AudioTower) preEncode(melIn []float32, frames int, s *audioScratch) []float32 {
	// The convolution wants the grid the other way round from the front end:
	// the frequency the faster index, where the mel comes mel-major. llama.cpp
	// opens its graph with exactly this transpose.
	freq, time := a.Cfg.MelBins, frames
	cur := s.grid[:freq*time]
	for m := 0; m < freq; m++ {
		row := melIn[m*frames:]
		for t := 0; t < time; t++ {
			cur[t*freq+m] = row[t]
		}
	}

	channels := 1
	for i := range a.W.Conv {
		c := a.W.Conv[i]
		outFreq, outTime := (freq-1)/2+1, (time-1)/2+1
		next := s.conv[i][:outFreq*outTime*c.Out]
		a.subsample(cur, next, c, freq, time, channels, outFreq, outTime)
		cur, freq, time, channels = next, outFreq, outTime, c.Out
	}

	// One point of the grid holds its channels already; a row of the tower is
	// the channels of the freq points that share a time, laid end to end with
	// the channel the faster index — which is what the points, running
	// frequency-fastest, already are. So this is a copy, and it is here rather
	// than folded into the caller because the projection wants a matrix.
	out := s.pre[:time*a.Cfg.Dim]
	a.W.InputProj.apply(cur[:time*freq*channels], s.held[:time*a.W.InputProj.Inputs], out, time)
	return out
}

// subsample is one 3x3 stride-2 convolution with its LayerNorm and its ReLU,
// over a point-major grid.
//
// The work is split over the output points rather than over the output
// channels: a point is where all three steps meet, so a core that owns one
// carries it from the gather to the ReLU without meeting anybody, and the
// kernel it reads — 147 kilobytes for the second convolution, five for the
// first — stays in its own cache the whole time.
func (a *AudioTower) subsample(in, out []float32, c AudioConv, freq, time, inCh, outFreq, outTime int) {
	points := outFreq * outTime
	taps := c.KW * c.KH * inCh

	nn.InParallel(points, points*taps*c.Out, func(first, last int) {
		patch := make([]float32, taps)
		for p := first; p < last; p++ {
			of, ot := p%outFreq, p/outFreq
			gather(patch, in, c, freq, time, inCh, of, ot)
			dst := out[p*c.Out : (p+1)*c.Out]

			if taps >= 64 {
				// Long rows: one dot product per output channel, which is what
				// the second convolution is — a thousand and fifty-two taps
				// against thirty-two channels.
				for o := range dst {
					dst[o] = nn.DotF32(patch, c.Packed[o*taps:(o+1)*taps])
				}
			} else {
				// Short rows and many channels, which is the first
				// convolution: nine taps against a hundred and twenty-eight.
				// A dot product of nine would spend its time being called, so
				// the sum runs the other way — each tap scaled into every
				// accumulator at once.
				for o := range dst {
					dst[o] = 0
				}
				for j, v := range patch {
					k := c.PackedT[j*c.Out:]
					for o := range dst {
						dst[o] += v * k[o]
					}
				}
			}
			layerNorm(dst, c.Norm, a.Cfg.Eps)
			for o, v := range dst {
				if v < 0 {
					dst[o] = 0
				}
			}
		}
	})
}

// gather writes the nine neighbourhoods of one output point into patch, in the
// order the packed kernel expects. Outside the grid it writes zeros, which is
// the symmetric padding.
func gather(patch, in []float32, c AudioConv, freq, time, inCh, of, ot int) {
	for kh := 0; kh < c.KH; kh++ {
		tt := 2*ot + kh - 1
		for kw := 0; kw < c.KW; kw++ {
			ff := 2*of + kw - 1
			row := patch[(kh*c.KW+kw)*inCh:][:inCh]
			if tt < 0 || tt >= time || ff < 0 || ff >= freq {
				for i := range row {
					row[i] = 0
				}
				continue
			}
			copy(row, in[(tt*freq+ff)*inCh:][:inCh])
		}
	}
}

// layerNorm normalizes one point of the grid across its own channels, then
// scales by the gain. It is a mean-and-variance normalization, not an RMS one;
// clip.cpp permutes its tensor twice to say the same thing.
func layerNorm(x, gain []float32, eps float32) {
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	var variance float32
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float32(len(x))
	scale := float32(1 / math.Sqrt(float64(variance)+float64(eps)))
	for i, v := range x {
		x[i] = (v - mean) * scale * gain[i]
	}
}

// apply computes Y = clamp(W * clamp(X)) for a batch of rows, in two passes.
//
// Two passes because the input's clamp belongs to the input and not to the
// product: written inside the loop over the output channels it is applied a
// thousand times to every value instead of once, which is a thousand times
// nothing per value and half a second per recording.
func (l AudioLinear) apply(x, held, y []float32, batch int) {
	lo, hi := l.Clamp.InMin, l.Clamp.InMax
	nn.InParallel(len(held), len(held), func(first, last int) {
		for i := first; i < last; i++ {
			held[i] = clampF32(x[i], lo, hi)
		}
	})
	outLo, outHi := l.Clamp.OutMin, l.Clamp.OutMax
	nn.InParallel(l.Outputs, batch*l.Inputs*l.Outputs, func(start, end int) {
		for b := 0; b < batch; b++ {
			row := held[b*l.Inputs : (b+1)*l.Inputs]
			dst := y[b*l.Outputs : (b+1)*l.Outputs]
			for o := start; o < end; o++ {
				dst[o] = clampF32(nn.DotF32(row, l.W[o*l.Inputs:(o+1)*l.Inputs]), outLo, outHi)
			}
		}
	})
}
