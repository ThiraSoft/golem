package qwen

// One block's attention, for a batch of consecutive positions.
//
// Twenty-eight blocks, all of one shape: sixteen query heads reading eight
// key-value ones, two queries to a key, a head a hundred and twenty-eight wide
// and every position seeing the whole prefix. There is no window to mask, no
// cache borrowed from another block, and no block that publishes keys and uses
// them as values.
//
// Each head is normed by its own weights before it is rotated — the queries by
// attn_q_norm, the keys by attn_k_norm — which is the one thing this block does
// that Gemma's does not. The value is neither normed nor rotated: position
// enters through the key alone, and llama.cpp hands the value projection
// straight to the attention.
//
// The batch is causal within itself: every key and value of the batch is in the
// cache before any score is computed, and a position reads the cache up to
// itself. That is why the two halves below are two sections and not one.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// Attention writes cfg.Dim floats per position into out: the attention output
// after the output projection, before the residual add.
func Attention(
	cfg *Config, bc BlockConfig, bw *BlockWeights,
	ropes []*nn.RoPETable, cache *Cache, s *Scratch,
	normed *nn.Batch, startPos int, out [][]float32,
) {
	batch := normed.Size
	lc := cache.Layers[bc.Index]
	// The scratch rows are as wide as the widest block; a narrower one writes
	// the head range it owns and reads it back by the same indices.
	q, qh := s.q[:batch], s.qh[:batch]
	k, v := s.k[:batch], s.v[:batch]

	// One section for the three projections and everything that follows them
	// head by head. The projections read the same input and do not depend on
	// one another, and a head's norm and rotation depend on that head alone —
	// so a core takes a head, projects it for every position of the batch,
	// norms it, rotates it and rounds it, and the block pays one barrier for
	// the lot.
	//
	// ggml multiplies the queries against an fp16 cache, and its matrix-vector
	// kernel converts the other operand to fp16 to do it. The same rounding has
	// to happen here, or the scores differ in the fourth digit — which is the
	// size of a real mistake.
	units := bc.Heads + bc.KVHeads
	nn.InParallel(units, units*bc.HeadDim*cfg.Dim*batch, func(start, end int) {
		for u := start; u < end; u++ {
			if u < bc.Heads {
				from, to := u*bc.HeadDim, (u+1)*bc.HeadDim
				bw.Q.MatVecRows(normed, q, from, to)
				for t := 0; t < batch; t++ {
					head := q[t][from:to]
					nn.RMSNormPlain(head, bw.QNorm, cfg.Eps)
					ropes[t].Apply(head[:bc.RoPEDims])
					for i, value := range head {
						qh[t][from+i] = nn.RoundHalf(value)
					}
				}
				continue
			}
			h := u - bc.Heads
			from, to := h*bc.HeadDim, (h+1)*bc.HeadDim
			bw.K.MatVecRows(normed, k, from, to)
			bw.V.MatVecRows(normed, v, from, to)
			for t := 0; t < batch; t++ {
				kh, vh := k[t][from:to], v[t][from:to]
				nn.RMSNormPlain(kh, bw.KNorm, cfg.Eps)
				ropes[t].Apply(kh[:bc.RoPEDims])
				lc.Store(startPos+t, h, kh, vh)
			}
		}
	})

	perKV := bc.Heads / bc.KVHeads
	// llama.cpp passes 1/sqrt(head_dim) into the softmax, which scales the
	// scores rather than the query. Gemma scales by one and lets its query norm
	// hold the scores in range; this model does not.
	scale := float32(1 / math.Sqrt(float64(bc.HeadDim)))

	// And one section for the attention itself, a head of a position to a core.
	// While an answer is being generated this is short; it is what grows with
	// the conversation, and what a prompt has most of.
	set := s.Batch(bc.Heads*bc.HeadDim, batch)
	nn.InParallel(batch*bc.Heads, 2*batch*bc.Heads*(startPos+batch)*bc.HeadDim, func(start, end int) {
		for u := start; u < end; u++ {
			t, h := u/bc.Heads, u%bc.Heads
			pos := startPos + t
			first, last := cache.Visible(bc, pos)
			n := last - first + 1
			scores := s.scores[t*s.maxHeads+h][:n]
			query := qh[t][h*bc.HeadDim : (h+1)*bc.HeadDim]
			kv := h / perKV

			for p := first; p <= last; p++ {
				scores[p-first] = scale * nn.DotF32(query, lc.Key(p, kv))
			}
			nn.SoftmaxInPlace(scores)
			// And again for the value product: the probabilities reach ggml's
			// kernel as fp16.
			for i, value := range scores {
				scores[i] = nn.RoundHalf(value)
			}

			dst := set.F[t][h*bc.HeadDim : (h+1)*bc.HeadDim]
			clear(dst)
			for p := first; p <= last; p++ {
				nn.Axpy(dst, lc.Value(p, kv), scores[p-first])
			}
			set.QuantizeColumnRange(t, h*bc.HeadDim, (h+1)*bc.HeadDim)
		}
	})

	bw.O.MatVecBatch(set, out)
}
