package qwen

// Binding the tensors to the geometry that named them.
//
// What a reader coming from gemma/weights.go will not find here, and should
// not go looking for: no per-layer embedding table and no projection for one,
// no post-attention or post-feed-forward norm — this model norms once before
// each half and puts the residual outside — no per-block output scalar, and no
// output head. The last is not an omission in the file: the checkpoint ties its
// head to its embedding, so the logits are token_embd read the other way round.
//
// What it has and Gemma does not is two norms of head width per block, one for
// the queries and one for the keys, applied per head before the rotation.

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

type BlockWeights struct {
	AttnNorm []float32 // before the attention
	FFNNorm  []float32 // before the feed forward
	QNorm    []float32 // HeadDim wide, applied to each query head
	KNorm    []float32 // HeadDim wide, applied to each key head

	Q, K, V, O     nn.Matrix
	Gate, Up, Down nn.Matrix
}

type Weights struct {
	TokenEmbd  nn.Matrix
	OutputNorm []float32
	Blocks     []BlockWeights
}

// Repack builds the interleaved form of every matrix a product reads, which
// nn/pack_q4_0.go describes. The embedding table is left alone: it is read a
// row at a time for the input, and the logit head reads it the same way.
//
// It is a second copy of the weights in memory and a few seconds of work at
// load time, and it is what makes a prompt a third faster. The matrices are
// independent, so the cores share them out.
func (w *Weights) Repack() {
	var all []*nn.Matrix
	for i := range w.Blocks {
		b := &w.Blocks[i]
		all = append(all, &b.Q, &b.K, &b.V, &b.O, &b.Gate, &b.Up, &b.Down)
	}

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

// LoadWeights binds every tensor the configuration says the model has, and
// checks each one against the shape the configuration expects. A matrix bound
// at the wrong width does not fail: it reads a neighbouring row and answers
// something plausible.
func LoadWeights(g *tensors.GGUF, cfg *Config) (*Weights, error) {
	w := &Weights{Blocks: make([]BlockWeights, len(cfg.Blocks))}

	var err error
	if w.TokenEmbd, err = matrix(g, "token_embd.weight"); err != nil {
		return nil, err
	}
	if w.OutputNorm, err = floats(g, "output_norm.weight"); err != nil {
		return nil, err
	}
	if len(w.OutputNorm) != cfg.Dim {
		return nil, fmt.Errorf("output_norm.weight is %d wide, not %d", len(w.OutputNorm), cfg.Dim)
	}
	// The head is the embedding read the other way round. A file that carries
	// its own head is a different model from this one, and quietly ignoring
	// two hundred megabytes of it would be the wrong kind of tolerant.
	if _, ok := g.Tensors["output.weight"]; ok {
		return nil, fmt.Errorf("this checkpoint has an output.weight; the engine assumes a tied head")
	}

	for i, bc := range cfg.Blocks {
		b := &w.Blocks[i]
		name := func(suffix string) string { return fmt.Sprintf("blk.%d.%s", i, suffix) }

		for _, bind := range []struct {
			dst  *[]float32
			name string
			want int
		}{
			{&b.AttnNorm, name("attn_norm.weight"), cfg.Dim},
			{&b.FFNNorm, name("ffn_norm.weight"), cfg.Dim},
			{&b.QNorm, name("attn_q_norm.weight"), bc.HeadDim},
			{&b.KNorm, name("attn_k_norm.weight"), bc.HeadDim},
		} {
			v, err := floats(g, bind.name)
			if err != nil {
				return nil, err
			}
			if len(v) != bind.want {
				return nil, fmt.Errorf("%s is %d wide, not %d", bind.name, len(v), bind.want)
			}
			*bind.dst = v
		}

		for _, bind := range []struct {
			dst        *nn.Matrix
			name       string
			rows, cols int
		}{
			{&b.Q, name("attn_q.weight"), bc.Heads * bc.HeadDim, cfg.Dim},
			{&b.K, name("attn_k.weight"), bc.KVHeads * bc.HeadDim, cfg.Dim},
			{&b.V, name("attn_v.weight"), bc.KVHeads * bc.HeadDim, cfg.Dim},
			{&b.O, name("attn_output.weight"), cfg.Dim, bc.Heads * bc.HeadDim},
			{&b.Gate, name("ffn_gate.weight"), bc.FFN, cfg.Dim},
			{&b.Up, name("ffn_up.weight"), bc.FFN, cfg.Dim},
			{&b.Down, name("ffn_down.weight"), cfg.Dim, bc.FFN},
		} {
			m, err := matrix(g, bind.name)
			if err != nil {
				return nil, err
			}
			if m.Rows != bind.rows || m.Cols != bind.cols {
				return nil, fmt.Errorf("%s is %d by %d, not %d by %d",
					bind.name, m.Rows, m.Cols, bind.rows, bind.cols)
			}
			*bind.dst = m
		}
	}
	return w, nil
}
