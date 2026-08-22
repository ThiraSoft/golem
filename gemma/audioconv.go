package gemma

// The conformer's convolution module: the part of a block that looks at a
// frame's neighbours rather than at the whole window.
//
// Six steps, and the fourth is the one with a trap in it. A norm; a pointwise
// projection to twice the width; a gated halving, the first half multiplied by
// the sigmoid of the second; a depthwise convolution of width five padded on
// the left alone, so frame t reads t-4 through t and nothing after it; a
// second norm; a SiLU; and a pointwise return to the width. llama.cpp writes
// that padding as a pad-then-roll around ggml_ssm_conv, which is a strange way
// to say "causal" and means exactly that. A symmetric padding here would run,
// would look plausible, and would let every frame hear two frames of its own
// future.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// convModule reads x — n positions of Cfg.Dim — and writes what the block adds
// to its residual into out.
func (a *AudioTower) convModule(b *AudioBlock, x, out []float32, n int, s *audioScratch) {
	cfg := a.Cfg
	dim := cfg.Dim
	const taps = 5

	norm := s.norm[:n*dim]
	copy(norm, x[:n*dim])
	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			nn.RMSNormPlain(norm[p*dim:(p+1)*dim], b.ConvPreNorm, cfg.Eps)
		}
	})

	wide := s.gated[:n*2*dim]
	b.ConvPW1.Apply(norm, s.wide, wide, n)

	// The gate. The two halves are the two halves of one row, not two
	// interleaved sets: the first dim values are the signal and the next dim
	// are what closes over it.
	gated := s.gate[:n*dim]
	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			row := wide[p*2*dim : (p+1)*2*dim]
			dst := gated[p*dim : (p+1)*dim]
			for i := 0; i < dim; i++ {
				dst[i] = row[i] / (1 + float32(math.Exp(float64(-row[dim+i]))))
			}
		}
	})

	// The depthwise convolution wants a channel's frames next to each other,
	// where everything else here wants a frame's channels next to each other.
	// One transpose, four zeros of history in front of it, and each channel is
	// a signal of its own.
	line := s.transposed[:(n+taps-1)*dim]
	for i := range line[:(taps-1)*dim] {
		line[i] = 0
	}
	nn.InParallel(dim, n*dim, func(first, last int) {
		for c := first; c < last; c++ {
			row := line[c*(n+taps-1)+taps-1:]
			for p := 0; p < n; p++ {
				row[p] = gated[p*dim+c]
			}
		}
	})
	nn.InParallel(dim, n*dim*taps, func(first, last int) {
		for c := first; c < last; c++ {
			row := line[c*(n+taps-1):]
			k := b.ConvDW[c*taps : (c+1)*taps]
			for p := 0; p < n; p++ {
				var sum float32
				for j := 0; j < taps; j++ {
					sum += k[j] * row[p+j]
				}
				out[p*dim+c] = sum
			}
		}
	})

	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			row := out[p*dim : (p+1)*dim]
			nn.RMSNormPlain(row, b.ConvInnerNorm, cfg.Eps)
			nn.SiLU(row)
		}
	})
	copy(gated, out[:n*dim])
	b.ConvPW2.Apply(gated, s.wide, out[:n*dim], n)
}
