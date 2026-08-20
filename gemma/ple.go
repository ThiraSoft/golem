package gemma

// The embedding, and the per-layer embeddings.
//
// Gemma 4 E2B gives every token a 1536-wide embedding like any transformer,
// and then a second, stranger thing: a 256-wide vector for each of the
// thirty-five blocks, which each block folds into the residual stream on its
// way out. Half of that vector is a table lookup — a row of a tensor of
// 262144 × 8960 in Q6_K, three quarters of a gigabyte — and half is a
// projection of the token's own embedding. The two halves are averaged with a
// 1/sqrt(2), which is what keeps the sum's variance where the model was
// trained to expect it.
//
// The 12B declares a per-layer width of zero and gets none of this.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// Embed writes the token's embedding, scaled by sqrt(Dim).
func Embed(cfg *Config, w *Weights, token int32, out []float32) {
	w.TokenEmbd.Row(int(token), out)
	scale := float32(math.Sqrt(float64(cfg.Dim)))
	for i := range out {
		out[i] *= scale
	}
}

// PerLayerInputs writes one 256-wide vector per block and per position of the
// batch, block-major within a position.
func PerLayerInputs(cfg *Config, w *Weights, s *Scratch, embedded *nn.Batch, tokens []int32, out [][]float32) {
	if cfg.PLEDim == 0 {
		return
	}
	width := cfg.PLEDim * len(cfg.Blocks)
	lookupScale := float32(math.Sqrt(float64(cfg.PLEDim)))
	projScale := float32(1 / math.Sqrt(float64(cfg.Dim)))
	inputScale := float32(1 / math.Sqrt2)

	// The projection is stored in bfloat16 and ggml rounds the activation to
	// bfloat16 to multiply by it, so the embedding is rounded here too: without
	// that the answer is out by a third of a percent, which is a hundred times
	// the gap everything else in this engine holds.
	rounded := s.BF16(len(tokens))
	nn.InParallel(len(tokens), len(tokens)*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			for i, v := range embedded.F[t] {
				rounded.F[t][i] = nn.RoundBF16(v)
			}
		}
	})
	proj := s.Batch(width, len(tokens))
	w.PLEProj.MatVecBatch(rounded, proj.F)

	nn.InParallel(len(tokens), len(tokens)*width*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			// The lookup half.
			w.PLEEmbd.Row(int(tokens[t]), out[t][:width])
			for i := 0; i < width; i++ {
				out[t][i] *= lookupScale
			}

			// The projected half, normalized one block at a time.
			for i := range proj.F[t] {
				proj.F[t][i] *= projScale
			}
			for b := range cfg.Blocks {
				nn.RMSNormPlain(proj.F[t][b*cfg.PLEDim:(b+1)*cfg.PLEDim], w.PLEProjNorm, cfg.Eps)
			}
			for i := 0; i < width; i++ {
				out[t][i] = (out[t][i] + proj.F[t][i]) * inputScale
			}
		}
	})
}
