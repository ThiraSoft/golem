package gemma

// One block's attention, for a batch of consecutive positions.
//
// Two geometries alternate through the model — four blocks that see 512
// positions with a rotation of base 10⁴, then one that sees everything with a
// base of 10⁶ and rotates only a quarter of each head — and from block 15 on,
// no block computes keys or values at all: they read what blocks 13 and 14
// left behind. None of that is a branch on the block number. Each block was
// asked what it is when the configuration was read, and answers here.
//
// The batch is causal within itself: every key and value of the batch is in the
// cache before any score is computed, and a position reads the cache up to
// itself. That is why the two halves below are two sections and not one.

import (
	"github.com/ThiraSoft/golem/nn"
)

// Attention writes cfg.Dim floats per position into out: the attention output
// after the output projection, before the post-attention norm.
//
// at says where each token of the batch goes: which conversation's cache it
// writes to, and at which position in it. A batch of one conversation is the
// ordinary case and every entry names the same cache; a batch drawn from
// several names several, and no two of them see each other.
func Attention(
	cfg *Config, bc BlockConfig, bw *BlockWeights,
	ropes []*nn.RoPETable, at []Place, s *Scratch,
	normed *nn.Batch, out [][]float32,
) {
	batch := normed.Size
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
	units := bc.Heads
	if bc.OwnsKV {
		units += bc.KVHeads
	}
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
			if !bc.ValueIsKey {
				bw.V.MatVecRows(normed, v, from, to)
			}
			for t := 0; t < batch; t++ {
				lc := at[t].Cache.Layers[bc.Index]
				kh, vh := k[t][from:to], v[t][from:to]
				if bc.ValueIsKey {
					// One projection for both, copied before the key is
					// normed and rotated: the value is neither.
					copy(vh, kh)
				}
				nn.RMSNormPlain(kh, bw.KNorm, cfg.Eps)
				// The value is normalized with no gain of its own, and is not
				// rotated: position enters through the key alone.
				nn.RMSNormPlain(vh, nil, cfg.Eps)
				ropes[t].Apply(kh[:bc.RoPEDims])
				lc.Store(at[t].Pos, h, kh, vh)
			}
		}
	})

	perKV := bc.Heads / bc.KVHeads

	// And one section for the attention itself, a head of a position to a core.
	// While an answer is being generated this is short; it is what grows with
	// the conversation, and what a prompt has most of.
	set := s.Batch(bc.Heads*bc.HeadDim, batch)
	nn.InParallel(batch*bc.Heads, 2*batch*bc.Heads*(deepest(at)+1)*bc.HeadDim, func(start, end int) {
		for u := start; u < end; u++ {
			t, h := u/bc.Heads, u%bc.Heads
			lc := at[t].Cache.Layers[bc.Index]
			pos := at[t].Pos
			first, last := at[t].Cache.Visible(bc, pos, at[t].Until)
			n := last - first + 1
			scores := s.scores[t*s.maxHeads+h][:n]
			query := qh[t][h*bc.HeadDim : (h+1)*bc.HeadDim]
			kv := h / perKV

			// No 1/sqrt(head_dim): Gemma 4 scales by one, and the query norm is
			// what keeps the scores in range.
			for p := first; p <= last; p++ {
				scores[p-first] = nn.DotF32(query, lc.Key(p, kv))
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

// deepest is the furthest into a conversation this batch reaches, which is
// what the attention of it costs. A token inside a picture reaches to the end
// of it, which is further than its own position.
func deepest(at []Place) int {
	n := 0
	for _, p := range at {
		if p.Pos > n {
			n = p.Pos
		}
		if p.Until > n {
			n = p.Until
		}
	}
	return n
}
