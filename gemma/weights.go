package gemma

// Binding the file's tensors to the shapes the engine computes with.
//
// Nothing is copied except the norm gains, which are small and read on every
// token. The weight matrices are views over the mapping: three gigabytes stay
// where they are, and the first token is produced before a reader would have
// finished loading them.

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

type BlockWeights struct {
	AttnNorm       []float32
	QNorm, KNorm   []float32
	Q, K, V, O     nn.Matrix
	PostAttnNorm   []float32
	FFNNorm        []float32
	Gate, Up, Down nn.Matrix
	PostFFWNorm    []float32
	InpGate, Proj  nn.Matrix
	PLENorm        []float32
	OutScale       float32
}

type Weights struct {
	TokenEmbd   nn.Matrix
	PLEEmbd     nn.Matrix
	PLEProj     nn.Matrix
	PLEProjNorm []float32
	OutputNorm  []float32
	RoPEFreqs   []float32
	Blocks      []BlockWeights
}

// matrix binds a two-dimensional tensor. GGUF writes the row length first, so
// Shape[0] counts inputs and Shape[1] outputs.
func matrix(g *tensors.GGUF, name string) (nn.Matrix, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nn.Matrix{}, fmt.Errorf("tensor %q is absent", name)
	}
	if len(t.Shape) != 2 {
		return nn.Matrix{}, fmt.Errorf("tensor %q has %d dimensions", name, len(t.Shape))
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		return nn.Matrix{}, fmt.Errorf("tensor %q is %s, which is not a weight format", name, t.DType)
	}
	return nn.Matrix{Data: t.Raw, Quant: q, Rows: t.Shape[1], Cols: t.Shape[0]}, nil
}

// floats copies an F32 tensor out of the mapping.
func floats(g *tensors.GGUF, name string) ([]float32, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q is absent", name)
	}
	if t.DType != "F32" {
		return nil, fmt.Errorf("tensor %q is %s, not F32", name, t.DType)
	}
	out := make([]float32, len(t.Raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Raw[4*i:]))
	}
	return out, nil
}

// LoadWeights binds every tensor the configuration says the model has.
func LoadWeights(g *tensors.GGUF, cfg *Config) (*Weights, error) {
	w := &Weights{Blocks: make([]BlockWeights, len(cfg.Blocks))}

	var err error
	if w.TokenEmbd, err = matrix(g, "token_embd.weight"); err != nil {
		return nil, err
	}
	if w.OutputNorm, err = floats(g, "output_norm.weight"); err != nil {
		return nil, err
	}
	// Optional: only the global blocks use it, and only some models have any.
	if f, err := floats(g, "rope_freqs.weight"); err == nil {
		w.RoPEFreqs = f
	}

	if cfg.PLEDim > 0 {
		if w.PLEEmbd, err = matrix(g, "per_layer_token_embd.weight"); err != nil {
			return nil, err
		}
		if w.PLEProj, err = matrix(g, "per_layer_model_proj.weight"); err != nil {
			return nil, err
		}
		if w.PLEProjNorm, err = floats(g, "per_layer_proj_norm.weight"); err != nil {
			return nil, err
		}
		if want := cfg.PLEDim * len(cfg.Blocks); w.PLEEmbd.Cols != want {
			return nil, fmt.Errorf("per-layer embedding is %d wide, expected %d", w.PLEEmbd.Cols, want)
		}
	}

	for i, bc := range cfg.Blocks {
		p := fmt.Sprintf("blk.%d.", i)
		b := BlockWeights{OutScale: 1}

		if b.AttnNorm, err = floats(g, p+"attn_norm.weight"); err != nil {
			return nil, err
		}
		if b.QNorm, err = floats(g, p+"attn_q_norm.weight"); err != nil {
			return nil, err
		}
		if b.Q, err = matrix(g, p+"attn_q.weight"); err != nil {
			return nil, err
		}
		if b.O, err = matrix(g, p+"attn_output.weight"); err != nil {
			return nil, err
		}
		if bc.OwnsKV {
			if b.K, err = matrix(g, p+"attn_k.weight"); err != nil {
				return nil, err
			}
			if b.V, err = matrix(g, p+"attn_v.weight"); err != nil {
				return nil, err
			}
			if b.KNorm, err = floats(g, p+"attn_k_norm.weight"); err != nil {
				return nil, err
			}
		}
		if b.PostAttnNorm, err = floats(g, p+"post_attention_norm.weight"); err != nil {
			return nil, err
		}
		if b.FFNNorm, err = floats(g, p+"ffn_norm.weight"); err != nil {
			return nil, err
		}
		if b.Gate, err = matrix(g, p+"ffn_gate.weight"); err != nil {
			return nil, err
		}
		if b.Up, err = matrix(g, p+"ffn_up.weight"); err != nil {
			return nil, err
		}
		if b.Down, err = matrix(g, p+"ffn_down.weight"); err != nil {
			return nil, err
		}
		if b.PostFFWNorm, err = floats(g, p+"post_ffw_norm.weight"); err != nil {
			return nil, err
		}
		if cfg.PLEDim > 0 {
			if b.InpGate, err = matrix(g, p+"inp_gate.weight"); err != nil {
				return nil, err
			}
			if b.Proj, err = matrix(g, p+"proj.weight"); err != nil {
				return nil, err
			}
			// llama.cpp calls this one per_layer_post_norm; the file calls it
			// post_norm, which reads like a block-final norm and is not one.
			if b.PLENorm, err = floats(g, p+"post_norm.weight"); err != nil {
				return nil, err
			}
		}
		if s, err := floats(g, p+"layer_output_scale.weight"); err == nil && len(s) == 1 {
			b.OutScale = s[0]
		}

		// The shapes have to agree with what the configuration said, or one of
		// the two is wrong and the products would run on the wrong bytes.
		if b.Q.Rows != bc.Heads*bc.HeadDim || b.Q.Cols != cfg.Dim {
			return nil, fmt.Errorf("block %d: query is %dx%d, expected %dx%d",
				i, b.Q.Rows, b.Q.Cols, bc.Heads*bc.HeadDim, cfg.Dim)
		}
		if b.O.Rows != cfg.Dim || b.O.Cols != bc.Heads*bc.HeadDim {
			return nil, fmt.Errorf("block %d: output projection is %dx%d", i, b.O.Rows, b.O.Cols)
		}
		if bc.OwnsKV && b.K.Rows != bc.KVHeads*bc.HeadDim {
			return nil, fmt.Errorf("block %d: key projection has %d rows, expected %d",
				i, b.K.Rows, bc.KVHeads*bc.HeadDim)
		}
		if b.Gate.Rows != bc.FFN || b.Down.Cols != bc.FFN {
			return nil, fmt.Errorf("block %d: feed forward is %d wide, expected %d", i, b.Gate.Rows, bc.FFN)
		}
		if len(b.QNorm) != bc.HeadDim {
			return nil, fmt.Errorf("block %d: query norm has %d entries for a head of %d", i, len(b.QNorm), bc.HeadDim)
		}
		w.Blocks[i] = b
	}
	return w, nil
}
