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
	ropes []*nn.RoPETable, at []Place, s *Scratch,
	xs [][]float32, ple [][]float32,
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
	Attention(cfg, bc, bw, ropes, at, s, normed, attn)
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

	if bc.MoE {
		moeHalf(cfg, bc, bw, s, xs, normed, batch)
	} else {
		denseHalf(cfg, bc, bw, s, xs, normed, batch)
	}

	// The per-layer embedding, folded in through its own residual. The
	// feed-forward buffer is free again by now, and is Dim wide.
	if cfg.PLEDim > 0 {
		ffn := s.ffn[:batch]
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

		bw.Proj.MatVecBatch(small, ffn)
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

// denseHalf is the feed-forward half of an ordinary block: one gated
// projection under ffn_norm, normed again on the way out, added to the stream.
//
// normed already holds the residual under ffn_norm, which the caller computed.
func denseHalf(cfg *Config, bc BlockConfig, bw *BlockWeights, s *Scratch, xs [][]float32, normed *nn.Batch, batch int) {
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
}

// moeHalf is the feed-forward half of a mixture block.
//
// Two branches leave the same residual: the dense feed forward through
// ffn_norm and post_ffw_norm_1, the experts through pre_ffw_norm_2 and
// post_ffw_norm_2. Their outputs are added, and the sum joins the stream —
// models/gemma4.cpp builds it in that order, and calls the result
// ffn_moe_combined.
//
// The dense branch is the shared expert. It reads exactly what denseHalf
// reads, under the same norm, which is why the shared expert needs no code of
// its own: it is the ordinary feed forward, and only its post-norm differs.
func moeHalf(cfg *Config, bc BlockConfig, bw *BlockWeights, s *Scratch, xs [][]float32, normed *nn.Batch, batch int) {
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
	shared := s.moeShared[:batch]
	bw.Down.MatVecBatch(gate, shared)

	// The expert branch, from the same residual under its own norm.
	expIn := s.ExpertBranch(batch)
	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			copy(expIn.F[t], s.resid[t])
			nn.RMSNormPlain(expIn.F[t], bw.PreFFWNorm2, cfg.Eps)
			expIn.QuantizeColumnRange(t, 0, cfg.Dim)
		}
	})
	experts := s.moeExperts[:batch]
	ExpertFFN(cfg, bw, s, expIn, experts)

	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			nn.RMSNormPlain(shared[t], bw.PostFFWNorm1, cfg.Eps)
			nn.RMSNormPlain(experts[t], bw.PostFFWNorm2, cfg.Eps)
			for i := range xs[t] {
				xs[t][i] = s.resid[t][i] + shared[t][i] + experts[t][i]
			}
		}
	})
}
