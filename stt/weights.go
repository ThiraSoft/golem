package stt

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

type Weights struct {
	Audio   [][]byte // Codebooks tables, each (CodeCard+1) x DModel, BF16
	Text    []byte   // (TextCard+1) x DModel, BF16
	Layers  []*Layer
	OutNorm []float32
	Head    nn.Linear
}

func LoadWeights(m *tensors.Model) (*Weights, error) {
	w := &Weights{
		Audio:  make([][]byte, Codebooks),
		Layers: make([]*Layer, NumLayers),
	}

	// Audio embedding tables: emb.{0..31}.weight [2049, 2048] BF16
	for q := 0; q < Codebooks; q++ {
		name := fmt.Sprintf("emb.%d.weight", q)
		t, err := m.Get(name)
		if err != nil {
			return nil, err
		}
		if len(t.Shape) != 2 || t.Shape[0] != CodeCard+1 || t.Shape[1] != DModel {
			return nil, fmt.Errorf("%s: shape %v, want [%d %d]", name, t.Shape, CodeCard+1, DModel)
		}
		if t.DType != "BF16" {
			return nil, fmt.Errorf("%s: dtype %s, want BF16", name, t.DType)
		}
		w.Audio[q] = t.Raw
	}

	// Text embedding table: text_emb.weight [8001, 2048] BF16
	textEmb, err := m.Get("text_emb.weight")
	if err != nil {
		return nil, err
	}
	if len(textEmb.Shape) != 2 || textEmb.Shape[0] != TextCard+1 || textEmb.Shape[1] != DModel {
		return nil, fmt.Errorf("text_emb.weight: shape %v, want [%d %d]", textEmb.Shape, TextCard+1, DModel)
	}
	if textEmb.DType != "BF16" {
		return nil, fmt.Errorf("text_emb.weight: dtype %s, want BF16", textEmb.DType)
	}
	w.Text = textEmb.Raw

	// Text head: text_linear.weight [8000, 2048] BF16
	textLin, err := m.Get("text_linear.weight")
	if err != nil {
		return nil, err
	}
	if len(textLin.Shape) != 2 || textLin.Shape[0] != TextCard || textLin.Shape[1] != DModel {
		return nil, fmt.Errorf("text_linear.weight: shape %v, want [%d %d]", textLin.Shape, TextCard, DModel)
	}
	if textLin.DType != "BF16" {
		return nil, fmt.Errorf("text_linear.weight: dtype %s, want BF16", textLin.DType)
	}
	w.Head = nn.Linear{Weights: textLin.Raw, Inputs: DModel, Outputs: TextCard}

	// Output norm: out_norm.alpha [1, 1, 2048] BF16
	outNorm, err := m.Get("out_norm.alpha")
	if err != nil {
		return nil, err
	}
	if outNorm.Elems() != DModel {
		return nil, fmt.Errorf("out_norm.alpha: %d elems, want %d", outNorm.Elems(), DModel)
	}
	if w.OutNorm, err = outNorm.F32(); err != nil {
		return nil, err
	}

	lin := func(name string, inputs, outputs int) (nn.Linear, error) {
		t, err := m.Get(name)
		if err != nil {
			return nn.Linear{}, err
		}
		if len(t.Shape) != 2 || t.Shape[0] != outputs || t.Shape[1] != inputs {
			return nn.Linear{}, fmt.Errorf("%s: shape %v, want [%d %d]", name, t.Shape, outputs, inputs)
		}
		if t.DType != "BF16" {
			return nn.Linear{}, fmt.Errorf("%s: dtype %s, want BF16", name, t.DType)
		}
		return nn.Linear{Weights: t.Raw, Inputs: inputs, Outputs: outputs}, nil
	}

	normAlpha := func(name string) ([]float32, error) {
		t, err := m.Get(name)
		if err != nil {
			return nil, err
		}
		if t.Elems() != DModel {
			return nil, fmt.Errorf("%s: %d elems, want %d", name, t.Elems(), DModel)
		}
		return t.F32()
	}

	for i := 0; i < NumLayers; i++ {
		prefix := fmt.Sprintf("transformer.layers.%d.", i)
		l := &Layer{}
		if l.Norm1, err = normAlpha(prefix + "norm1.alpha"); err != nil {
			return nil, err
		}
		if l.Norm2, err = normAlpha(prefix + "norm2.alpha"); err != nil {
			return nil, err
		}
		if l.InProj, err = lin(prefix+"self_attn.in_proj_weight", DModel, 3*DModel); err != nil {
			return nil, err
		}
		if l.OutProj, err = lin(prefix+"self_attn.out_proj.weight", DModel, DModel); err != nil {
			return nil, err
		}
		if l.GateIn, err = lin(prefix+"gating.linear_in.weight", DModel, 2*DimFF); err != nil {
			return nil, err
		}
		if l.GateOut, err = lin(prefix+"gating.linear_out.weight", DimFF, DModel); err != nil {
			return nil, err
		}
		w.Layers[i] = l
	}

	return w, nil
}
