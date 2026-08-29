package qwen35

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// VisionNorm is one layer normalization of the tower: a gain and a bias, where
// every norm of the text model has a gain alone. clip.cpp calls it
// NORM_TYPE_NORMAL, which is ggml_norm — the mean subtracted, not an RMS.
type VisionNorm struct{ Gain, Bias []float32 }

// VisionLinear is a projection with a bias, which every product in this tower
// has and no product in the text model does.
type VisionLinear struct {
	W    nn.Matrix
	Bias []float32
}

// VisionBlock is one block's weights, in the order the pass reads them.
type VisionBlock struct {
	LN1 VisionNorm
	QKV VisionLinear // fused, 3*Dim out
	O   VisionLinear
	LN2 VisionNorm
	Up  VisionLinear
	Dn  VisionLinear
}

// VisionWeights is the whole projector.
type VisionWeights struct {
	// PatchA and PatchB are the two halves of a convolution whose temporal
	// depth is two. llama.cpp splits the Conv3d into two Conv2d tensors and
	// sums them; a still image is the same frame twice, so both meet the same
	// pixels. Each is [Patch, Patch, 3, Dim] in the file's order.
	PatchA, PatchB nn.Matrix
	PatchBias      []float32

	// PosEmbd is the learned table, PosSide by PosSide of Dim, stored as F32.
	PosEmbd []float32

	Blocks []VisionBlock
	PostLN VisionNorm

	// MM0 and MM2 are the merger: Merge*Merge patches concatenated, through
	// the first with a GELU after it, then the second onto the text model's
	// width. The file names them mm.0 and mm.2 — there is no mm.1.
	MM0, MM2 VisionLinear
}

// visionMatrix binds a two-dimensional tensor and checks its shape. GGUF
// writes the row length first, so Shape[0] counts inputs and Shape[1] outputs.
// A matrix bound at the wrong width does not fail: it reads a neighbouring row
// and answers something plausible.
func visionMatrix(g *tensors.GGUF, name string, inputs, outputs int) (nn.Matrix, error) {
	t, err := g.Get(name)
	if err != nil {
		return nn.Matrix{}, err
	}
	if len(t.Shape) != 2 || t.Shape[0] != inputs || t.Shape[1] != outputs {
		return nn.Matrix{}, fmt.Errorf("qwen35: %s is %v, wanted %d by %d", name, t.Shape, inputs, outputs)
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		return nn.Matrix{}, fmt.Errorf("qwen35: %s is %s, which nn does not read", name, t.DType)
	}
	return nn.Matrix{Data: t.Raw, Quant: q, Rows: t.Shape[1], Cols: t.Shape[0]}, nil
}

// visionFloats copies an F32 tensor out of the mapping and checks its length.
func visionFloats(g *tensors.GGUF, name string, want int) ([]float32, error) {
	t, err := g.Get(name)
	if err != nil {
		return nil, err
	}
	out, err := t.F32()
	if err != nil {
		return nil, err
	}
	if want > 0 && len(out) != want {
		return nil, fmt.Errorf("qwen35: %s holds %d floats, wanted %d", name, len(out), want)
	}
	return out, nil
}

func visionNorm(g *tensors.GGUF, prefix string, width int) (VisionNorm, error) {
	gain, err := visionFloats(g, prefix+".weight", width)
	if err != nil {
		return VisionNorm{}, err
	}
	bias, err := visionFloats(g, prefix+".bias", width)
	if err != nil {
		return VisionNorm{}, err
	}
	return VisionNorm{Gain: gain, Bias: bias}, nil
}

func visionLinear(g *tensors.GGUF, prefix string, inputs, outputs int) (VisionLinear, error) {
	w, err := visionMatrix(g, prefix+".weight", inputs, outputs)
	if err != nil {
		return VisionLinear{}, err
	}
	bias, err := visionFloats(g, prefix+".bias", outputs)
	if err != nil {
		return VisionLinear{}, err
	}
	return VisionLinear{W: w, Bias: bias}, nil
}

// LoadVisionWeights binds every tensor the tower computes with, checking each
// against the geometry the configuration read.
func LoadVisionWeights(g *tensors.GGUF, cfg *VisionConfig) (*VisionWeights, error) {
	w := &VisionWeights{Blocks: make([]VisionBlock, cfg.Blocks)}

	// The patch convolutions are four-dimensional in the file — width, height,
	// channel, output — and are read as a matrix of Dim rows over
	// Patch*Patch*3 inputs, which is what a gathered patch is a vector of.
	taps := cfg.Patch * cfg.Patch * 3
	for _, c := range []struct {
		name string
		into *nn.Matrix
	}{
		{"v.patch_embd.weight", &w.PatchA},
		{"v.patch_embd.weight.1", &w.PatchB},
	} {
		t, err := g.Get(c.name)
		if err != nil {
			return nil, err
		}
		if len(t.Shape) != 4 || t.Shape[0] != cfg.Patch || t.Shape[1] != cfg.Patch ||
			t.Shape[2] != 3 || t.Shape[3] != cfg.Dim {
			return nil, fmt.Errorf("qwen35: %s is %v, wanted %d by %d by 3 by %d",
				c.name, t.Shape, cfg.Patch, cfg.Patch, cfg.Dim)
		}
		q, ok := nn.QuantOf(t.DType)
		if !ok {
			return nil, fmt.Errorf("qwen35: %s is %s, which nn does not read", c.name, t.DType)
		}
		*c.into = nn.Matrix{Data: t.Raw, Quant: q, Rows: cfg.Dim, Cols: taps}
	}
	var err error
	if w.PatchBias, err = visionFloats(g, "v.patch_embd.bias", cfg.Dim); err != nil {
		return nil, err
	}
	if w.PosEmbd, err = visionFloats(g, "v.position_embd.weight", cfg.Dim*cfg.PosSide*cfg.PosSide); err != nil {
		return nil, err
	}

	for i := range w.Blocks {
		p := fmt.Sprintf("v.blk.%d.", i)
		b := &w.Blocks[i]
		if b.LN1, err = visionNorm(g, p+"ln1", cfg.Dim); err != nil {
			return nil, err
		}
		if b.QKV, err = visionLinear(g, p+"attn_qkv", cfg.Dim, 3*cfg.Dim); err != nil {
			return nil, err
		}
		if b.O, err = visionLinear(g, p+"attn_out", cfg.Dim, cfg.Dim); err != nil {
			return nil, err
		}
		if b.LN2, err = visionNorm(g, p+"ln2", cfg.Dim); err != nil {
			return nil, err
		}
		if b.Up, err = visionLinear(g, p+"ffn_up", cfg.Dim, cfg.FFN); err != nil {
			return nil, err
		}
		if b.Dn, err = visionLinear(g, p+"ffn_down", cfg.FFN, cfg.Dim); err != nil {
			return nil, err
		}
	}

	if w.PostLN, err = visionNorm(g, "v.post_ln", cfg.Dim); err != nil {
		return nil, err
	}
	merged := cfg.Dim * cfg.Merge * cfg.Merge
	if w.MM0, err = visionLinear(g, "mm.0", merged, merged); err != nil {
		return nil, err
	}
	if w.MM2, err = visionLinear(g, "mm.2", merged, cfg.ProjDim); err != nil {
		return nil, err
	}
	return w, nil
}
