package gemma

// Binding the file's tensors to the shapes the engine computes with.
//
// The weight matrices are views over the mapping, and the norm gains — small,
// and read on every token — are the only thing copied out of it. Then Repack
// builds a second layout of the matrices a product reads, which is the one
// exception and is paid for in nn/pack_q4_0.go: a third of a second at load and
// a gigabyte of memory on E2B, against a prompt read half again as fast.

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

	// A mixture block only, and zero on every other. The router reads the
	// residual rather than either branch's normed input, and its scale vector
	// stands in for the gain of a norm that has none.
	Router                     nn.Matrix
	RouterScale                []float32
	PreFFWNorm2                []float32
	PostFFWNorm1, PostFFWNorm2 []float32
	// GateUpExps holds both halves of one product: an expert's first
	// ExpertFFN rows are its gate, the next ExpertFFN its up. The file fuses
	// them and the reference splits them back into two views; keeping the
	// fusion is one product over 1408 rows rather than two over 704, on an
	// input read once.
	GateUpExps, DownExps ExpertStack
	// DownScale is one scalar per expert, multiplying that expert's whole
	// output. llama.cpp applies it just before the routing weight, so this
	// engine folds it into that weight.
	DownScale []float32
}

// A stack of expert matrices, kept as the one three-dimensional tensor the
// file stores rather than split at load time.
//
// Splitting would mean thirty blocks times three hundred and eighty-four
// matrices, and would buy nothing: only eight experts of a hundred and
// twenty-eight are read per position, so Repack — which pays for itself on a
// matrix every token reads whole — would double the resident weights to speed
// up six percent of the rows. At goes to the bytes instead.
type ExpertStack struct {
	Data       []byte
	Quant      nn.Quant
	Rows, Cols int // one expert's shape: Rows outputs, each reading Cols inputs
	Count      int
}

// At is one expert's matrix, as a view over the mapping.
func (e ExpertStack) At(i int) nn.Matrix {
	m := nn.Matrix{Quant: e.Quant, Rows: e.Rows, Cols: e.Cols}
	size := m.RowBytes() * e.Rows
	m.Data = e.Data[i*size : (i+1)*size]
	return m
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

// Repack builds the interleaved form of every matrix a product reads, which
// nn/pack_q4_0.go describes. The embedding tables are left alone: they are read
// a row at a time and never multiplied.
//
// The expert stacks are left out as well, for a different reason than the
// embedding tables: they are multiplied, but only eight of a hundred and
// twenty-eight per position, so a second copy of them would cost more memory
// than the product it saves.
//
// It is a second copy of the weights in memory and a few seconds of work at
// load time, and it is what makes a prompt a third faster. The matrices are
// independent, so the cores share them out.
func (w *Weights) Repack() {
	var all []*nn.Matrix
	for i := range w.Blocks {
		b := &w.Blocks[i]
		all = append(all, &b.Q, &b.K, &b.V, &b.O, &b.Gate, &b.Up, &b.Down, &b.InpGate, &b.Proj)
	}
	all = append(all, &w.PLEProj)

	nn.InParallel(len(all), 1<<30, func(first, last int) {
		for i := first; i < last; i++ {
			all[i].Repack()
		}
	})
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

// experts binds a three-dimensional tensor. GGUF writes the fastest dimension
// first, so the shape is inputs, outputs, experts.
func experts(g *tensors.GGUF, name string) (ExpertStack, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return ExpertStack{}, fmt.Errorf("tensor %q is absent", name)
	}
	if len(t.Shape) != 3 {
		return ExpertStack{}, fmt.Errorf("tensor %q has %d dimensions, expected 3", name, len(t.Shape))
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		return ExpertStack{}, fmt.Errorf("tensor %q is %s, which is not a weight format", name, t.DType)
	}
	e := ExpertStack{Data: t.Raw, Quant: q, Rows: t.Shape[1], Cols: t.Shape[0], Count: t.Shape[2]}
	row := nn.Matrix{Quant: q, Cols: e.Cols}
	if want := row.RowBytes() * e.Rows * e.Count; want != len(t.Raw) {
		return ExpertStack{}, fmt.Errorf("tensor %q holds %d bytes for %d experts of %dx%d, expected %d",
			name, len(t.Raw), e.Count, e.Rows, e.Cols, want)
	}
	return e, nil
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
			if !bc.ValueIsKey {
				if b.V, err = matrix(g, p+"attn_v.weight"); err != nil {
					return nil, err
				}
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
		if bc.MoE {
			if b.Router, err = matrix(g, p+"ffn_gate_inp.weight"); err != nil {
				return nil, err
			}
			if b.RouterScale, err = floats(g, p+"ffn_gate_inp.scale"); err != nil {
				return nil, err
			}
			if b.PreFFWNorm2, err = floats(g, p+"pre_ffw_norm_2.weight"); err != nil {
				return nil, err
			}
			if b.PostFFWNorm1, err = floats(g, p+"post_ffw_norm_1.weight"); err != nil {
				return nil, err
			}
			if b.PostFFWNorm2, err = floats(g, p+"post_ffw_norm_2.weight"); err != nil {
				return nil, err
			}
			if b.GateUpExps, err = experts(g, p+"ffn_gate_up_exps.weight"); err != nil {
				return nil, err
			}
			if b.DownExps, err = experts(g, p+"ffn_down_exps.weight"); err != nil {
				return nil, err
			}
			if b.DownScale, err = floats(g, p+"ffn_down_exps.scale"); err != nil {
				return nil, err
			}
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
		if bc.MoE {
			if b.Router.Rows != cfg.Experts || b.Router.Cols != cfg.Dim {
				return nil, fmt.Errorf("block %d: router is %dx%d, expected %dx%d",
					i, b.Router.Rows, b.Router.Cols, cfg.Experts, cfg.Dim)
			}
			if len(b.RouterScale) != cfg.Dim {
				return nil, fmt.Errorf("block %d: the router scale has %d entries for a width of %d",
					i, len(b.RouterScale), cfg.Dim)
			}
			if len(b.DownScale) != cfg.Experts {
				return nil, fmt.Errorf("block %d: the down scale has %d entries for %d experts",
					i, len(b.DownScale), cfg.Experts)
			}
			for _, e := range []struct {
				name       string
				stack      ExpertStack
				rows, cols int
			}{
				{"gate and up", b.GateUpExps, 2 * cfg.ExpertFFN, cfg.Dim},
				{"down", b.DownExps, cfg.Dim, cfg.ExpertFFN},
			} {
				if e.stack.Count != cfg.Experts || e.stack.Rows != e.rows || e.stack.Cols != e.cols {
					return nil, fmt.Errorf("block %d: %s experts are %d of %dx%d, expected %d of %dx%d",
						i, e.name, e.stack.Count, e.stack.Rows, e.stack.Cols, cfg.Experts, e.rows, e.cols)
				}
			}
		}
		if len(b.QNorm) != bc.HeadDim {
			return nil, fmt.Errorf("block %d: query norm has %d entries for a head of %d", i, len(b.QNorm), bc.HeadDim)
		}
		w.Blocks[i] = b
	}
	return w, nil
}
