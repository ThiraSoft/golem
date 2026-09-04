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
	Head    nn.Matrix
	Quant   nn.Quant // what the sixteen blocks were converted to
}

// LoadWeights reads the trunk, converting its projections to `quant` as it
// goes. nn.BF16 keeps the file's own format, which is what the tests that hold
// this code against PyTorch ask for; nn.Q4_0 is what a transcriber runs, and
// stt/quantize.go says why.
//
// The embedding tables and the logit head stay bfloat16 whatever is asked. The
// tables are read one row at a time — thirty-three rows a frame, not a product
// — and the head is 33 MiB against the blocks' 1.64 GiB, so neither is on the
// path that the bus decides.
func LoadWeights(m *tensors.Model, quant nn.Quant) (*Weights, error) {
	if quant != nn.BF16 && quant != nn.Q4_0 {
		return nil, fmt.Errorf("stt: the trunk is read as bfloat16 or as Q4_0, not as %s", quant)
	}
	w := &Weights{
		Audio:  make([][]byte, Codebooks),
		Layers: make([]*Layer, NumLayers),
		Quant:  quant,
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
	w.Head = nn.Matrix{Data: textLin.Raw, Quant: nn.BF16, Rows: TextCard, Cols: DModel}

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

	// Every projection goes through here, so the conversion is written once
	// and no matrix can be forgotten by a later reader adding a fifth one.
	lin := func(name string, inputs, outputs int) (nn.Matrix, error) {
		t, err := m.Get(name)
		if err != nil {
			return nn.Matrix{}, err
		}
		if len(t.Shape) != 2 || t.Shape[0] != outputs || t.Shape[1] != inputs {
			return nn.Matrix{}, fmt.Errorf("%s: shape %v, want [%d %d]", name, t.Shape, outputs, inputs)
		}
		if t.DType != "BF16" {
			return nn.Matrix{}, fmt.Errorf("%s: dtype %s, want BF16", name, t.DType)
		}
		out := nn.Matrix{Data: t.Raw, Quant: nn.BF16, Rows: outputs, Cols: inputs}
		if quant == nn.Q4_0 {
			out.Data = quantizeQ4_0(t.Raw, outputs, inputs)
			out.Quant = nn.Q4_0
			// The interleaved form of the same bytes, which is what the widest
			// kernel reads. nn/pack_q4_0.go says what it buys.
			out.Repack()
		}
		return out, nil
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
