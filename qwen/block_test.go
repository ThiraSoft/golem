package qwen

import (
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// The feed forward on its own, from the reference's own normed input.
//
// This is where the non-linearity is settled. nn.SwiGLURange reproduces both
// halves of ggml_vec_swiglu_f32: the polynomial exponential, and the fusion of
// the gate's activation with the up projection into one pass. The gap it
// leaves is what this test prints.
func TestFeedForwardMatchesReference(t *testing.T) {
	f := loadFixture(t, "layers")
	cfg, w := loadForTest(t)
	w.Repack()
	s := NewScratchFor(cfg, w)

	for _, il := range []int{0, 1, 14, 27} {
		bc := cfg.Blocks[il]
		bw := &w.Blocks[il]

		// The last column of each, which is the last token for every block and
		// the only one for block 27.
		normed := s.Batch(cfg.Dim, 1)
		normed.Set(0, f.lastColumn(t, "ffn_norm-"+itoa(il)))

		gate := s.Batch(bc.FFN, 1)
		up := make([]float32, bc.FFN)
		bw.Gate.MatVecRows(normed, gate.F, 0, bc.FFN)
		bw.Up.MatVecRows(normed, [][]float32{up}, 0, bc.FFN)
		nn.SwiGLURange(gate.F[0], up, 0, bc.FFN)
		gate.QuantizeColumnRange(0, 0, bc.FFN)

		out := [][]float32{make([]float32, cfg.Dim)}
		bw.Down.MatVecBatch(gate, out)

		compareRelative(t, "ffn_out-"+itoa(il), out[0], f.lastColumn(t, "ffn_out-"+itoa(il)), 1e-3)
	}
}

// One whole block, from the reference's own input, against every waypoint it
// passes through. A failure says which half.
func TestBlockMatchesReference(t *testing.T) {
	f := loadFixture(t, "layers")
	cfg, w := loadForTest(t)
	w.Repack()
	s := NewScratchFor(cfg, w)

	for _, il := range []int{0, 1, 14, 27} {
		bc := cfg.Blocks[il]
		bw := &w.Blocks[il]
		cache := NewCache(cfg)

		// A block is fed the reference's own output from the block before it,
		// so a failure here is this block's. Block 0 reads the embedding,
		// which llama.cpp does not record — the full-stack test covers it.
		if il == 0 {
			continue
		}
		source := "l_out-" + itoa(il-1)

		// Run every position, so the cache is filled the way generation fills
		// it, then compare the last one.
		last := len(f.Tokens) - 1
		xs := make([][]float32, len(f.Tokens))
		for p := range xs {
			xs[p] = append([]float32(nil), f.column(t, source, p)...)
		}
		s.Reserve(len(xs))
		Block(cfg, bc, bw, s.RoPE(bc, Run(cache, 0, len(xs))), Run(cache, 0, len(xs)), s, xs)

		compareRelative(t, "l_out-"+itoa(il), xs[last], f.lastColumn(t, "l_out-"+itoa(il)), 2e-3)
	}
}

// The value between the halves: the attention's output projection added to the
// block's input, before the feed-forward norm. llama.cpp calls it ffn_inp, and
// it is the first place the attention as a whole can be checked.
func TestAttentionOutputMatchesReference(t *testing.T) {
	f := loadFixture(t, "layers")
	cfg, w := loadForTest(t)
	w.Repack()
	s := NewScratchFor(cfg, w)

	for _, il := range []int{1, 14, 27} {
		bc := cfg.Blocks[il]
		bw := &w.Blocks[il]
		cache := NewCache(cfg)
		last := len(f.Tokens) - 1
		s.Reserve(len(f.Tokens))

		xs := make([][]float32, len(f.Tokens))
		for p := range xs {
			xs[p] = append([]float32(nil), f.column(t, "l_out-"+itoa(il-1), p)...)
		}

		// The first half of Block, by hand, stopping at the residual.
		normed := s.Batch(cfg.Dim, len(xs))
		for p := range xs {
			copy(normed.F[p], xs[p])
			nn.RMSNormPlain(normed.F[p], bw.AttnNorm, cfg.Eps)
			normed.QuantizeColumnRange(p, 0, cfg.Dim)
		}
		attn := make([][]float32, len(xs))
		for p := range attn {
			attn[p] = make([]float32, cfg.Dim)
		}
		s.Reserve(len(xs))
		Attention(cfg, bc, bw, s.RoPE(bc, Run(cache, 0, len(xs))), Run(cache, 0, len(xs)), s, normed, attn)
		for i := range xs[last] {
			xs[last][i] += attn[last][i]
		}

		compareRelative(t, "ffn_inp-"+itoa(il), xs[last], f.lastColumn(t, "ffn_inp-"+itoa(il)), 1e-3)
	}
}
