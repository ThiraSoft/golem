package qwen35

import (
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// ForwardFullAttnToken processes one token through a full attention block.
//
// Qwen3.8-27B full attention:
//   - 24 query heads, 4 KV heads (six queries per key)
//   - HeadDim 256, RoPE over the first 64 elements of each head
//   - the query projection carries the output gate with it: its 12288 outputs
//     are twenty-four pairs of [query(256) | gate(256)], one pair per head, not
//     a block of queries followed by a block of gates. llama.cpp reads the two
//     as strided views of the same tensor (models/qwen35.cpp, build_layer_attn),
//     and reading them as halves silently pairs head h's queries with head
//     h/2's gate.
//   - QNorm and KNorm run per head, before RoPE
//   - the mix is scaled by sigmoid(gate) before the output projection
func ForwardFullAttnToken(
	cfg *Config, bc BlockConfig, bw *BlockWeights,
	cache *BlockCache, rope *nn.RoPETable, at Place,
	x []float32, out []float32, scratch *Scratch,
) {
	// The cache index. The rotation reads the other three axes, and only it
	// does — place.go says why the two are not one number.
	pos := at.Pos
	headDim := bc.HeadDim
	heads := bc.Heads
	kvHeads := bc.KVHeads
	qDim := heads * headDim
	kvDim := kvHeads * headDim
	pair := headDim * 2 // one head's [query | gate]

	qFull := scratch.qFull[:qDim*2]
	k := scratch.k[:kvDim]
	v := scratch.v[:kvDim]

	bw.Q.MatVec(scratch.batchX, qFull)
	bw.K.MatVec(scratch.batchX, k)
	bw.V.MatVec(scratch.batchX, v)

	// Query norm, then RoPE, head by head inside the interleaving.
	if rope != nil {
		rope.PrepareMulti(bc.RoPEDims, at.at(), bc.RoPEBase, cfg.RoPESections, nil)
	}
	for h := 0; h < heads; h++ {
		head := qFull[h*pair : h*pair+headDim]
		nn.RMSNormPlain(head, bw.QNorm, cfg.Eps)
		if rope != nil {
			rope.Apply(head[:bc.RoPEDims])
		}
	}
	for h := 0; h < kvHeads; h++ {
		head := k[h*headDim : (h+1)*headDim]
		nn.RMSNormPlain(head, bw.KNorm, cfg.Eps)
		if rope != nil {
			rope.Apply(head[:bc.RoPEDims])
		}
	}

	kvStride := kvDim
	copy(cache.Keys[pos*kvStride:(pos+1)*kvStride], k)
	copy(cache.Values[pos*kvStride:(pos+1)*kvStride], v)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	headsPerKV := heads / kvHeads
	attnOut := scratch.o[:qDim]

	nn.InParallel(heads, heads*(pos+1)*headDim, func(first, last int) {
		scores := make([]float32, pos+1)
		for h := first; h < last; h++ {
			qHead := qFull[h*pair : h*pair+headDim]
			kvHeadIdx := h / headsPerKV
			oHead := attnOut[h*headDim : (h+1)*headDim]

			maxScore := float32(math.Inf(-1))
			for p := 0; p <= pos; p++ {
				base := p*kvStride + kvHeadIdx*headDim
				dot := nn.DotF32(qHead, cache.Keys[base:base+headDim]) * scale
				scores[p] = dot
				if dot > maxScore {
					maxScore = dot
				}
			}

			var sumExp float32
			for p := 0; p <= pos; p++ {
				e := float32(math.Exp(float64(scores[p] - maxScore)))
				scores[p] = e
				sumExp += e
			}
			invSum := 1.0 / sumExp

			clear(oHead)
			for p := 0; p <= pos; p++ {
				weight := scores[p] * invSum
				base := p*kvStride + kvHeadIdx*headDim
				nn.AxpyFull(oHead, cache.Values[base:base+headDim], weight)
			}

			// The gate sits right after this head's queries.
			gate := qFull[h*pair+headDim : (h+1)*pair]
			for i := 0; i < headDim; i++ {
				oHead[i] *= float32(1.0 / (1.0 + math.Exp(float64(-gate[i]))))
			}
		}
	})

	load(scratch.batchAttn, attnOut)
	calib(bc.Index, "o", attnOut)
	bw.O.MatVec(scratch.batchAttn, out)
}
