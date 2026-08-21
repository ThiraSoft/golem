package qwen

import (
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// The waypoints inside one position, in order, so that a failure says which
// step went wrong rather than that the block did.
//
// llama.cpp records the queries and keys on both sides of the rotation, which
// is what makes this test worth writing in this order: if Qcur_normed matches
// and Qcur does not, the projection and the per-head norm are right and the
// rotation is not.
func TestAttentionWaypoints(t *testing.T) {
	f := loadFixture(t, "layers")
	cfg, w := loadForTest(t)
	w.Repack()
	s := NewScratchFor(cfg, w)

	for _, il := range []int{0, 1, 14, 27} {
		bc := cfg.Blocks[il]
		bw := &w.Blocks[il]
		cache := NewCache(cfg)
		out := [][]float32{make([]float32, cfg.Dim)}
		normed := s.Batch(cfg.Dim, 1)
		const pos = 3

		// Fill the cache up to the position under test, from the reference's
		// own inputs, so that a divergence at pos is this position's.
		for p := 0; p < pos; p++ {
			normed.Set(0, f.column(t, "attn_norm-"+itoa(il), p))
			Attention(cfg, bc, bw, s.RoPE(bc, p, 1), cache, s, normed, p, out)
		}
		normed.Set(0, f.column(t, "attn_norm-"+itoa(il), pos))
		Attention(cfg, bc, bw, s.RoPE(bc, pos, 1), cache, s, normed, pos, out)

		// After the rotation, which is what the engine leaves in the scratch.
		compare(t, "Qcur-"+itoa(il), s.Q(bc), f.heads(t, "Qcur-"+itoa(il), pos), 2e-4)
		compare(t, "Kcur-"+itoa(il), s.K(bc), f.heads(t, "Kcur-"+itoa(il), pos), 2e-4)
		// The value is neither normed nor rotated: position enters through the
		// key alone.
		compare(t, "Vcur-"+itoa(il), s.V(bc), f.heads(t, "Vcur-"+itoa(il), pos), 2e-4)
	}
}

// The projection and the per-head norm on their own, before any rotation. This
// is the finest waypoint the recording offers, and the first place to look
// when Qcur is wrong.
func TestQueryAndKeyNormsMatchBeforeRotation(t *testing.T) {
	f := loadFixture(t, "layers")
	cfg, w := loadForTest(t)
	w.Repack()
	s := NewScratchFor(cfg, w)

	for _, il := range []int{0, 27} {
		bc := cfg.Blocks[il]
		bw := &w.Blocks[il]
		const pos = 2

		normed := s.Batch(cfg.Dim, 1)
		normed.Set(0, f.column(t, "attn_norm-"+itoa(il), pos))

		// Project and norm by hand, stopping short of the rotation.
		q := make([]float32, bc.Heads*bc.HeadDim)
		k := make([]float32, bc.KVHeads*bc.HeadDim)
		bw.Q.MatVecRows(normed, [][]float32{q}, 0, bc.Heads*bc.HeadDim)
		bw.K.MatVecRows(normed, [][]float32{k}, 0, bc.KVHeads*bc.HeadDim)
		for h := 0; h < bc.Heads; h++ {
			nn.RMSNormPlain(q[h*bc.HeadDim:(h+1)*bc.HeadDim], bw.QNorm, cfg.Eps)
		}
		for h := 0; h < bc.KVHeads; h++ {
			nn.RMSNormPlain(k[h*bc.HeadDim:(h+1)*bc.HeadDim], bw.KNorm, cfg.Eps)
		}

		compare(t, "Qcur_normed-"+itoa(il), q, f.heads(t, "Qcur_normed-"+itoa(il), pos), 2e-4)
		compare(t, "Kcur_normed-"+itoa(il), k, f.heads(t, "Kcur_normed-"+itoa(il), pos), 2e-4)
	}
}
