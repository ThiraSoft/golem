package qwen

// One block of the model, for a batch of consecutive positions.
//
// Plain pre-norm, which is the shape most transformers have and Gemma does not:
// norm, attend, add; norm, feed forward, add. Two norms, and the residual
// outside both halves.
//
// Gemma's block norms on both sides of each half — before the attention and
// after it, before the feed forward and after it — with the residual add
// outside all four, and then folds in a per-layer embedding and scales the
// whole thing. None of that is here, and the file it should be compared
// against is gemma/block.go.
//
// llama.cpp names the value between the halves ffn_inp: the attention's output
// projection added to the block's input. It is the first waypoint at which the
// attention can be checked, because qwen3.cpp records nothing between the
// heads and the projection.

import "github.com/ThiraSoft/golem/nn"

// perPosition is what a pass over one position costs, in the units InParallel
// counts: a norm, a residual add and a quantization come to a few dozen
// operations for every element of the stream.
//
// Guessing low is not a small mistake. At four, a batch of sixty-four
// positions of this model came to 262144, just under the threshold below which
// InParallel declines to split — so three passes a block, eighty-four in a
// prompt, ran on one core while seven spun. That was 29% of the reading.
const perPosition = 24

func Block(
	cfg *Config, bc BlockConfig, bw *BlockWeights,
	ropes []*nn.RoPETable, at []Place, s *Scratch,
	xs [][]float32,
) {
	batch := len(xs)

	// Attention, pre-normed.
	//
	// The passes that touch one position at a time are spread over the cores
	// too, by position: at a batch of one they are a rounding error, and at the
	// batch a prompt offers they would otherwise be the one thing running on a
	// single core while the others waited.
	normed := s.Batch(cfg.Dim, batch)
	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			copy(normed.F[t], xs[t])
			nn.RMSNormPlain(normed.F[t], bw.AttnNorm, cfg.Eps)
			normed.QuantizeColumnRange(t, 0, cfg.Dim)
		}
	})

	attn := s.attn[:batch]
	Attention(cfg, bc, bw, ropes, at, s, normed, attn)

	// The residual, with no norm between it and the projection — this is
	// llama.cpp's ffn_inp, and it is written back over the input.
	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			for i := range xs[t] {
				xs[t][i] += attn[t][i]
			}
			copy(normed.F[t], xs[t])
			nn.RMSNormPlain(normed.F[t], bw.FFNNorm, cfg.Eps)
			normed.QuantizeColumnRange(t, 0, cfg.Dim)
		}
	})

	// The whole first half of the feed forward in one section: a core computes
	// its rows of the gate and of the up projection for every position of the
	// batch, activates them, multiplies them together and quantizes the result
	// before the section ends. Written as five passes over the same values, the
	// four cheap ones would run on one core while the others waited at four
	// barriers.
	//
	// SiLU on the gate, times the up projection, in one pass — ggml's
	// swiglu, where Gemma has a tabulated GELU.
	gate := s.Batch(bc.FFN, batch)
	up := s.up[:batch]
	blocks := bc.FFN / nn.QuantBlock
	nn.InParallel(blocks, 2*bc.FFN*cfg.Dim*batch, func(first, last int) {
		from, to := first*nn.QuantBlock, last*nn.QuantBlock
		bw.Gate.MatVecRows(normed, gate.F, from, to)
		bw.Up.MatVecRows(normed, up, from, to)
		for t := 0; t < batch; t++ {
			nn.SwiGLURange(gate.F[t], up[t], from, to)
			gate.QuantizeColumnRange(t, from, to)
		}
	})

	ffn := s.ffn[:batch]
	bw.Down.MatVecBatch(gate, ffn)

	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			for i := range xs[t] {
				xs[t][i] += ffn[t][i]
			}
		}
	})
}
