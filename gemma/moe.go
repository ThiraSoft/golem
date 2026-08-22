package gemma

// The mixture of experts.
//
// Gemma 4's mixture is not the usual one. The shared expert is the ordinary
// dense feed forward, and the experts run beside it rather than after it: two
// branches leave the same residual, each through its own pair of norms, and
// their outputs are added. Block computes the dense half; this file is the
// other one.
//
// The router is the part worth reading twice. It reads the residual — the
// stream after the attention, before either branch's norm — normalizes it
// without a gain, scales it by one over the square root of the width, and
// multiplies elementwise by a vector the file carries in place of that missing
// gain. models/gemma4.cpp builds exactly that, under the comment "router
// operates on attn_out, not cur". Routing on the branch's normed input instead
// gives a model that answers fluently and wrongly, which nothing announces.

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// Route chooses each position's experts and their weights.
//
// resid is the residual after the attention half, one row per position. ids
// and weights are ExpertsUsed wide per position and are written; the logits
// land in the scratch, where the reference tests read them.
//
// The whole batch goes through the router's matrix at once, because that is
// one product of Experts rows rather than one per position. The choosing that
// follows is per position and costs nothing beside it.
func Route(cfg *Config, bw *BlockWeights, s *Scratch, resid [][]float32, ids [][]int32, weights [][]float32) {
	batch := len(resid)
	in := s.RouterIn(batch)
	scale := float32(1 / math.Sqrt(float64(cfg.Dim)))
	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			copy(in.F[t], resid[t])
			nn.RMSNormPlain(in.F[t], nil, cfg.Eps)
			for i := range in.F[t] {
				in.F[t][i] *= scale * bw.RouterScale[i]
			}
			in.QuantizeColumnRange(t, 0, cfg.Dim)
		}
	})
	bw.Router.MatVecBatch(in, s.expLogits[:batch])

	for t := 0; t < batch; t++ {
		chooseExperts(cfg, s.expLogits[t], ids[t], weights[t])
	}
}

// chooseExperts takes the largest few logits and turns them into weights that
// sum to one.
//
// By selection rather than by sorting: eight of a hundred and twenty-eight,
// once per block per token, and a sort would allocate. A tie goes to the lower
// index, so that two runs of the same model choose the same experts and a
// comparison against a recording means something.
func chooseExperts(cfg *Config, logits []float32, ids []int32, weights []float32) {
	for k := 0; k < cfg.ExpertsUsed; k++ {
		best := int32(-1)
		for e := 0; e < cfg.Experts; e++ {
			if taken(ids[:k], int32(e)) {
				continue
			}
			if best < 0 || logits[e] > logits[best] {
				best = int32(e)
			}
		}
		ids[k] = best
	}

	// The softmax is over the chosen few alone, renormalized. The largest is
	// subtracted first, as everywhere else, so that a large logit does not
	// become an infinity on its way through the exponential.
	top := logits[ids[0]]
	total := float32(0)
	for k, id := range ids {
		weights[k] = float32(math.Exp(float64(logits[id] - top)))
		total += weights[k]
	}
	for k := range weights {
		weights[k] /= total
	}
}

func taken(ids []int32, e int32) bool {
	for _, id := range ids {
		if id == e {
			return true
		}
	}
	return false
}

// ExpertFFN is the expert branch, for a batch of positions.
//
// in is the branch's normed input, quantized, one column per position; out is
// written. The routing reads s.resid, which the block has already filled with
// the residual: the router's input is not this branch's input, and taking both
// as arguments would let a caller pass the same thing twice.
//
// The work is spread over the positions rather than over the rows of one
// expert. Each position reads a different eight matrices, so a section per
// expert would be a barrier per expert — a hundred and twenty-eight of them a
// block, in the worst case, for products of seven hundred and four rows. At a
// batch of one, one core does the arithmetic of eight small matrices, which is
// little beside the attention it follows.
func ExpertFFN(cfg *Config, bw *BlockWeights, s *Scratch, in *nn.Batch, out [][]float32) {
	batch := len(out)
	Route(cfg, bw, s, s.resid[:batch], s.expIDs[:batch], s.expWeights[:batch])

	work := batch * cfg.ExpertsUsed * cfg.ExpertFFN * cfg.Dim * 2
	nn.InParallel(batch, work, func(first, last int) {
		for t := first; t < last; t++ {
			// One column of the batch, on its own, because an expert's matrix
			// is read for this position and no other.
			one, mid := s.ExpertIn(t), s.ExpertMid(t)
			copy(one.F[0], in.F[t])
			one.QuantizeColumnRange(0, 0, cfg.Dim)

			gateUp, down := s.expGateUp[t], s.expDown[t]
			gate, up := gateUp[:cfg.ExpertFFN], gateUp[cfg.ExpertFFN:]
			for i := range out[t] {
				out[t][i] = 0
			}
			for k, id := range s.expIDs[t] {
				// One product for both halves. MatVecRows rather than
				// MatVecBatch: this is already inside a section, and
				// MatVecBatch would open another.
				gu := bw.GateUpExps.At(int(id))
				gu.MatVecRows(one, s.expGateUpRow[t], 0, gu.Rows)

				nn.GELUTable(gate)
				for i := range gate {
					mid.F[0][i] = gate[i] * up[i]
				}
				mid.QuantizeColumnRange(0, 0, cfg.ExpertFFN)

				d := bw.DownExps.At(int(id))
				d.MatVecRows(mid, s.expDownRow[t], 0, d.Rows)

				// The per-expert scale multiplies the whole of this expert's
				// output, and so does the routing weight; one multiplication
				// does both.
				nn.Axpy(out[t], down, s.expWeights[t][k]*bw.DownScale[id])
			}
		}
	})
}
