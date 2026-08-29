package qwen35

import (
	"github.com/ThiraSoft/golem/nn"
)

// Block runs one layer of Qwen3.8: a pre-norm, a mixer that is either the
// gated delta net or full attention, a residual, a post-attention norm, a
// SwiGLU feed forward, and a second residual. Both block kinds share the feed
// forward, and only the mixer differs.
func Block(
	cfg *Config, bc BlockConfig, bw *BlockWeights,
	cache *BlockCache, rope *nn.RoPETable, at Place,
	x []float32, scratch *Scratch,
) {
	dim := cfg.Dim
	normed := scratch.normed[:dim]
	subOut := scratch.subOut[:dim]

	copy(normed, x)
	nn.RMSNormPlain(normed, bw.AttnNorm, cfg.Eps)
	scratch.SetInput(normed)

	if bc.Type == BlockFullAttn {
		ForwardFullAttnToken(cfg, bc, bw, cache, rope, at, normed, subOut, scratch)
	} else {
		ForwardSSMToken(cfg, bc, bw, cache, normed, subOut, scratch)
	}

	for i := 0; i < dim; i++ {
		x[i] += subOut[i]
	}

	copy(normed, x)
	nn.RMSNormPlain(normed, bw.FFNNorm, cfg.Eps)
	scratch.SetInput(normed)

	gate := scratch.batchFFN.F[0][:bc.FFN]
	up := scratch.ffnUp[:bc.FFN]
	bw.Gate.MatVec(scratch.batchX, gate)
	bw.Up.MatVec(scratch.batchX, up)

	nn.SiLUGGMLRange(gate, 0, bc.FFN)
	for i := 0; i < bc.FFN; i++ {
		gate[i] *= up[i]
	}
	quantize(scratch.batchFFN)

	ffnOut := scratch.ffnDown[:dim]
	bw.Down.MatVec(scratch.batchFFN, ffnOut)

	for i := 0; i < dim; i++ {
		x[i] += ffnOut[i]
	}
}
