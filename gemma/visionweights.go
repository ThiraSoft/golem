package gemma

// Binding the projector file's tensors.
//
// Everything here is bfloat16 or F32 and none of it is repacked: the tower
// runs once per image, not once per token, and the layout that pays for itself
// in the text model would buy nothing here.
//
// Each weight in the file is shadowed by four scalars — input_min, input_max,
// output_min, output_max — the clipping ranges the model was quantized under.
// llama.cpp applies them to the projection alone (build_mm in
// models/gemma4v.cpp), and so does this file.

import (
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// Clamp is one Gemma4ClippableLinear's two ranges.
type Clamp struct{ InMin, InMax, OutMin, OutMax float32 }

type VisionBlockWeights struct {
	LN1, LN2                  []float32
	QNorm, KNorm              []float32
	Q, K, V, O                nn.Linear
	PostAttnNorm, PostFFNNorm []float32
	Gate, Up, Down            nn.Linear
}

type VisionWeights struct {
	PatchEmbd  []float32
	PosX, PosY []float32
	Positions  int
	Blocks     []VisionBlockWeights
	Proj       nn.Linear
	ProjClamp  Clamp
	StdBias    []float32
	StdScale   []float32
}

// linear binds a BF16 matrix as a projection with no bias. GGUF writes the row
// length first, so Shape[0] counts inputs and Shape[1] outputs — the same
// convention gemma/weights.go reads.
func linear(g *tensors.GGUF, name string) (nn.Linear, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nn.Linear{}, fmt.Errorf("tensor %q is absent", name)
	}
	if t.DType != "BF16" {
		return nn.Linear{}, fmt.Errorf("tensor %q is %s; the projector is expected to be BF16", name, t.DType)
	}
	if len(t.Shape) != 2 {
		return nn.Linear{}, fmt.Errorf("tensor %q has %d dimensions", name, len(t.Shape))
	}
	return nn.Linear{Weights: t.Raw, Inputs: t.Shape[0], Outputs: t.Shape[1]}, nil
}

// scalar reads one of the QAT range tensors, which are one-element F32. A
// range the file does not carry is no range at all, which is what the infinite
// fallback says.
func scalar(g *tensors.GGUF, name string, fallback float32) float32 {
	v, err := floats(g, name)
	if err != nil || len(v) != 1 {
		return fallback
	}
	return v[0]
}

func clampOf(g *tensors.GGUF, weight string) Clamp {
	base := weight[:len(weight)-len(".weight")]
	inf := float32(math.Inf(1))
	return Clamp{
		InMin:  scalar(g, base+".input_min", -inf),
		InMax:  scalar(g, base+".input_max", inf),
		OutMin: scalar(g, base+".output_min", -inf),
		OutMax: scalar(g, base+".output_max", inf),
	}
}

// LoadVisionWeights binds every tensor the tower computes with.
func LoadVisionWeights(g *tensors.GGUF, cfg *VisionConfig) (*VisionWeights, error) {
	w := &VisionWeights{Blocks: make([]VisionBlockWeights, cfg.Blocks)}

	var err error
	if w.PatchEmbd, err = floats(g, "v.patch_embd.weight"); err != nil {
		return nil, err
	}
	if n := cfg.PatchSize * cfg.PatchSize * 3 * cfg.Dim; len(w.PatchEmbd) != n {
		return nil, fmt.Errorf("the patch embedding holds %d floats, expected %d", len(w.PatchEmbd), n)
	}

	// One tensor holding two tables: the x lookup, then the y lookup.
	pos, err := floats(g, "v.position_embd.weight")
	if err != nil {
		return nil, err
	}
	if len(pos)%(2*cfg.Dim) != 0 {
		return nil, fmt.Errorf("the position table holds %d floats, which is not two tables of %d-wide rows", len(pos), cfg.Dim)
	}
	w.Positions = len(pos) / (2 * cfg.Dim)
	w.PosX = pos[:w.Positions*cfg.Dim]
	w.PosY = pos[w.Positions*cfg.Dim:]

	for i := range w.Blocks {
		p := fmt.Sprintf("v.blk.%d.", i)
		b := VisionBlockWeights{}
		for _, f := range []struct {
			name string
			dst  *[]float32
		}{
			{p + "ln1.weight", &b.LN1},
			{p + "ln2.weight", &b.LN2},
			{p + "attn_q_norm.weight", &b.QNorm},
			{p + "attn_k_norm.weight", &b.KNorm},
			{p + "attn_post_norm.weight", &b.PostAttnNorm},
			{p + "ffn_post_norm.weight", &b.PostFFNNorm},
		} {
			if *f.dst, err = floats(g, f.name); err != nil {
				return nil, err
			}
		}
		for _, f := range []struct {
			name string
			dst  *nn.Linear
		}{
			{p + "attn_q.weight", &b.Q},
			{p + "attn_k.weight", &b.K},
			{p + "attn_v.weight", &b.V},
			{p + "attn_out.weight", &b.O},
			{p + "ffn_gate.weight", &b.Gate},
			{p + "ffn_up.weight", &b.Up},
			{p + "ffn_down.weight", &b.Down},
		} {
			if *f.dst, err = linear(g, f.name); err != nil {
				return nil, err
			}
		}
		if b.Q.Inputs != cfg.Dim || b.Q.Outputs != cfg.Dim {
			return nil, fmt.Errorf("block %d: query is %dx%d, expected %dx%d",
				i, b.Q.Outputs, b.Q.Inputs, cfg.Dim, cfg.Dim)
		}
		if b.Gate.Outputs != cfg.FFN || b.Down.Inputs != cfg.FFN {
			return nil, fmt.Errorf("block %d: feed forward is %d wide, expected %d", i, b.Gate.Outputs, cfg.FFN)
		}
		if len(b.QNorm) != cfg.HeadDim {
			return nil, fmt.Errorf("block %d: query norm has %d entries for a head of %d", i, len(b.QNorm), cfg.HeadDim)
		}
		w.Blocks[i] = b
	}

	if w.Proj, err = linear(g, "mm.input_projection.weight"); err != nil {
		return nil, err
	}
	if w.Proj.Inputs != cfg.Dim || w.Proj.Outputs != cfg.ProjDim {
		return nil, fmt.Errorf("the projection is %dx%d, expected %dx%d",
			w.Proj.Outputs, w.Proj.Inputs, cfg.ProjDim, cfg.Dim)
	}
	w.ProjClamp = clampOf(g, "mm.input_projection.weight")

	// Optional, and llama.cpp treats them as a pair: both or neither. E2B's
	// projector carries neither, and the standardisation is then not a step.
	bias, errB := floats(g, "v.std_bias")
	scale, errS := floats(g, "v.std_scale")
	if errB == nil && errS == nil {
		w.StdBias, w.StdScale = bias, scale
	}
	return w, nil
}
