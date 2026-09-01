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

	// What the activations of each site meet before they reach a Golem matrix:
	// the reciprocal of the per-column scale the weights were quantized under,
	// with the rotation's sign flips folded in. Empty for every other format.
	// nn/golem.go says why the scheme is split this way.
	PreQKV, PreO, PreGateUp, PreDown []float32

	// HadGroup is the width of that rotation, zero when there is none.
	HadGroup int
}

type Weights struct {
	TokenEmbd  nn.Matrix
	OutputNorm []float32
	Blocks     []BlockWeights

	// PreHead is the head's own vector, and the head is the embedding table:
	// the logit product meets it like any other site, and the input path undoes
	// it a row at a time. Nil for a table stored plain.
	PreHead []float32

	// HadGroup is the width of the rotation a Golem checkpoint's activations go
	// through, and zero for every other format.
	HadGroup int

	// ActBF16 says the activations must be rounded to bfloat16 before they
	// meet these weights. ggml converts an activation to whatever the weight's
	// dot product wants, and for a bfloat16 weight that is bfloat16 — so a
	// float32 activation is more precision than the reference has.
	ActBF16 bool
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

	// A .golem file carries one vector a site. Any other file has none, and
	// the blocks are left with nil ones — which is what the products check.
	if group, err := g.Uint32("golem.hadamard_group"); err == nil {
		w.HadGroup = int(group)
		for i := range cfg.Blocks {
			b := &w.Blocks[i]
			// A vector belongs to a matrix, not to a block: a checkpoint that
			// left some tensors alone has vectors for the others and none for
			// those, and a matrix that is not Golem must not meet one.
			for _, bind := range []struct {
				dst  *[]float32
				site string
				want int
				m    nn.Matrix
			}{
				{&b.PreQKV, "qkv", b.Q.Cols, b.Q},
				{&b.PreO, "o", b.O.Cols, b.O},
				{&b.PreGateUp, "gateup", b.Gate.Cols, b.Gate},
				{&b.PreDown, "down", b.Down.Cols, b.Down},
			} {
				if !bind.m.Quant.Golem() {
					continue
				}
				name := fmt.Sprintf("blk.%d.%s.pre", i, bind.site)
				v, err := floats(g, name)
				if err != nil {
					return nil, err
				}
				if len(v) != bind.want {
					return nil, fmt.Errorf("%s is %d wide, not %d", name, len(v), bind.want)
				}
				*bind.dst = v
			}
			b.HadGroup = w.HadGroup
		}
		if v, err := floats(g, "output.pre"); err == nil && w.TokenEmbd.Quant.Golem() {
			if len(v) != w.TokenEmbd.Cols {
				return nil, fmt.Errorf("output.pre is %d wide, not %d", len(v), w.TokenEmbd.Cols)
			}
			w.PreHead = v
		}
	}

	// Every matrix in a checkpoint has the same format, and the first one
	// answers for all of them.
	w.ActBF16 = w.Blocks[0].Q.Quant == nn.BF16
	return w, nil
}
