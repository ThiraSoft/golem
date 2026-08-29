package qwen35

import (
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// tinyTower is a tower of one block at a geometry small enough to reason
// about: eight wide, two heads of four, a feed forward of sixteen. The weights
// are deterministic and mean nothing; what these tests ask is whether the pass
// is a function of its input, not whether it is the right function. That is
// the reference test's question.
func tinyTower() (*VisionConfig, *VisionWeights) {
	cfg := &VisionConfig{
		Blocks: 1, Dim: 8, Heads: 2, HeadDim: 4, FFN: 16,
		Patch: 2, Merge: 2, ProjDim: 8, PosSide: 2, Eps: 1e-6,
		MinTokens: 1, MaxTokens: 4096,
	}
	f16 := func(rows, cols int, seed float32) nn.Matrix {
		half := make([]uint16, rows*cols)
		for i := range half {
			half[i] = nn.Half(float32(i%11)*0.05 - 0.25 + seed)
		}
		return nn.Matrix{
			Data:  unsafe.Slice((*byte)(unsafe.Pointer(&half[0])), len(half)*2),
			Quant: nn.F16, Rows: rows, Cols: cols,
		}
	}
	floats := func(n int, seed float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(i%5)*0.1 + seed
		}
		return out
	}
	w := &VisionWeights{
		Blocks: []VisionBlock{{
			LN1: VisionNorm{Gain: floats(8, 1), Bias: floats(8, 0)},
			QKV: VisionLinear{W: f16(24, 8, 0.1), Bias: floats(24, 0)},
			O:   VisionLinear{W: f16(8, 8, 0.2), Bias: floats(8, 0)},
			LN2: VisionNorm{Gain: floats(8, 1), Bias: floats(8, 0)},
			Up:  VisionLinear{W: f16(16, 8, 0.15), Bias: floats(16, 0)},
			Dn:  VisionLinear{W: f16(8, 16, 0.05), Bias: floats(8, 0)},
		}},
		PostLN: VisionNorm{Gain: floats(8, 1), Bias: floats(8, 0)},
		MM0:    VisionLinear{W: f16(32, 32, 0.1), Bias: floats(32, 0)},
		MM2:    VisionLinear{W: f16(8, 32, 0.1), Bias: floats(8, 0)},
	}
	return cfg, w
}
