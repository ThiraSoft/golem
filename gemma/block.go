package gemma

// One block of the model, for a batch of consecutive positions.
//
// Gemma norms on both sides of each half — before the attention and after it,
// before the feed forward and after it — with the residual add outside all
// four. Then, in E2B only, the block's own slice of the token's per-layer
// embedding is gated, projected and added, and the whole thing is multiplied by
// a single scalar the file carries per block.
//
// The names in the file are not the names in the paper. post_attention_norm is
// the norm *after* the attention and inside the residual; post_ffw_norm the
// same for the feed forward; and post_norm, which sounds like the block's last
// word, is the norm on the per-layer branch alone.

import "github.com/ThiraSoft/golem/nn"

func Block(
	cfg *Config, bc BlockConfig, bw *BlockWeights,
	ropes []*nn.RoPETable, cache *Cache, s *Scratch,
	xs [][]float32, ple [][]float32, startPos int,
) {
	batch := len(xs)

	// Attention, normed on both sides.
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
	Attention(cfg, bc, bw, ropes, cache, s, normed, startPos, attn)
	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			nn.RMSNormPlain(attn[t], bw.PostAttnNorm, cfg.Eps)
			for i := range s.resid[t] {
				s.resid[t][i] = xs[t][i] + attn[t][i]
			}
			copy(normed.F[t], s.resid[t])
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
	gate := s.Batch(bc.FFN, batch)
	up := s.up[:batch]
	blocks := bc.FFN / nn.QuantBlock
	nn.InParallel(blocks, 2*bc.FFN*cfg.Dim*batch, func(first, last int) {
		from, to := first*nn.QuantBlock, last*nn.QuantBlock
		bw.Gate.MatVecRows(normed, gate.F, from, to)
		bw.Up.MatVecRows(normed, up, from, to)
		for t := 0; t < batch; t++ {
			nn.GELUTable(gate.F[t][from:to])
			for i := from; i < to; i++ {
				gate.F[t][i] *= up[t][i]
			}
			gate.QuantizeColumnRange(t, from, to)
		}
	})

	ffn := s.ffn[:batch]
	bw.Down.MatVecBatch(gate, ffn)

	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			nn.RMSNormPlain(ffn[t], bw.PostFFWNorm, cfg.Eps)
			for i := range xs[t] {
				xs[t][i] = s.resid[t][i] + ffn[t][i]
			}
		}
	})

	// The per-layer embedding, folded in through its own residual.
	if cfg.PLEDim > 0 {
		small := s.Batch(cfg.PLEDim, batch)
		nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
			for t := first; t < last; t++ {
				copy(s.pe[t], xs[t])
				copy(normed.F[t], xs[t])
				normed.QuantizeColumnRange(t, 0, cfg.Dim)
			}
		})
		bw.InpGate.MatVecBatch(normed, small.F)
		nn.InParallel(batch, batch*cfg.PLEDim*perPosition, func(first, last int) {
			for t := first; t < last; t++ {
				nn.GELUTable(small.F[t])
				for i := range small.F[t] {
					small.F[t][i] *= ple[t][i]
				}
				small.QuantizeColumnRange(t, 0, cfg.PLEDim)
			}
		})

		bw.Proj.MatVecBatch(small, ffn) // ffn is free again, and is Dim wide
		nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
			for t := first; t < last; t++ {
				nn.RMSNormPlain(ffn[t], bw.PLENorm, cfg.Eps)
				for i := range xs[t] {
					xs[t][i] = s.pe[t][i] + ffn[t][i]
				}
			}
		})
	}

	for _, x := range xs {
		for i := range x {
			x[i] *= bw.OutScale
		}
	}
}

// perPosition is what a pass over one position costs, in the units InParallel
// counts: a norm, a residual add and a quantization come to a few dozen
// operations for every element of the stream.
const perPosition = 24
