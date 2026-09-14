// Package nomic runs nomic-embed-text-v2-moe, the embedder GGUF calls
// nomic-bert-moe: text in, one vector out.
//
// It is a BERT, and that makes it unlike every other engine here. Nothing is
// generated, so there is no cache and no logit head: a text goes through once,
// every position sees every other, and what comes out of the last block is
// averaged into one vector. The blocks are post-norm — the residual is added
// and then the sum is normed, twice a block — where Gemma and Qwen norm before.
// Every projection carries a bias. Position enters through a NeoX rotation of
// the queries and keys, as in the decoders, and not through a learned table as
// in the original BERT.
//
// Every odd block's feed forward is a mixture: eight experts, the two a router
// likes best answering, weighted by the router's softmax as it stood — not
// renormalized over the two. An expert is the dense feed forward without its
// biases.
//
// The reference is llama.cpp, src/models/bert.cpp, which builds this graph for
// four architectures; the branches taken for this one are the ones followed
// here.
package nomic

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/ugm"
	"github.com/ThiraSoft/golem/vk"
)

// Pooling is how the positions of a text become one vector: llama.cpp's
// llama_pooling_type, which the file declares.
type Pooling int

const (
	PoolNone Pooling = 0
	PoolMean Pooling = 1
	PoolCLS  Pooling = 2
	PoolLast Pooling = 3
)

type Config struct {
	Dim, Heads, HeadDim  int
	Blocks, FF           int
	Experts, ExpertsUsed int
	// MoEEvery says which blocks are mixtures: block i is when i%MoEEvery is
	// 1. Zero means none.
	MoEEvery int
	Eps      float32
	RoPEBase float64
	// Context is the most positions a text may have, which is what the
	// checkpoint was trained on.
	Context int
	Pooling Pooling
}

// Mixture says whether block i is a mixture of experts.
func (c *Config) Mixture(i int) bool { return c.MoEEvery > 0 && i%c.MoEEvery == 1 }

func LoadConfig(g *tensors.GGUF) (*Config, error) {
	arch, err := g.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "nomic-bert-moe" {
		return nil, fmt.Errorf("nomic: the file is %q, not nomic-bert-moe", arch)
	}
	u := func(key string, dst *int) {
		if err != nil {
			return
		}
		var v uint32
		v, err = g.Uint32(arch + "." + key)
		*dst = int(v)
	}
	c := &Config{}
	u("embedding_length", &c.Dim)
	u("attention.head_count", &c.Heads)
	u("block_count", &c.Blocks)
	u("feed_forward_length", &c.FF)
	u("expert_count", &c.Experts)
	u("expert_used_count", &c.ExpertsUsed)
	u("moe_every_n_layers", &c.MoEEvery)
	u("context_length", &c.Context)
	var pooling int
	u("pooling_type", &pooling)
	if err != nil {
		return nil, err
	}
	c.Pooling = Pooling(pooling)
	if c.Eps, err = g.Float32(arch + ".attention.layer_norm_epsilon"); err != nil {
		return nil, err
	}
	base, err := g.Float32(arch + ".rope.freq_base")
	if err != nil {
		return nil, err
	}
	c.RoPEBase = float64(base)
	if causal, err := g.Bool(arch + ".attention.causal"); err == nil && causal {
		return nil, fmt.Errorf("nomic: the file declares a causal attention, which this encoder is not")
	}
	if c.Heads == 0 || c.Dim%c.Heads != 0 {
		return nil, fmt.Errorf("nomic: %d heads do not divide a width of %d", c.Heads, c.Dim)
	}
	c.HeadDim = c.Dim / c.Heads
	switch c.Pooling {
	case PoolMean, PoolCLS, PoolLast:
	default:
		return nil, fmt.Errorf("nomic: pooling type %d is not implemented; mean, cls and last are", c.Pooling)
	}
	return c, nil
}

type BlockWeights struct {
	QKV, O           nn.Matrix // with their biases
	AttnNorm         nn.LayerNorm
	OutNorm          nn.LayerNorm
	Up, Down         nn.Matrix // a dense block's, with their biases
	Router           nn.Matrix // a mixture's: Experts rows
	UpExps, DownExps []nn.Matrix
}

type Weights struct {
	TokenEmbd nn.Matrix
	// TypeEmbd is the first row of the token-type table. llama.cpp hard-codes
	// every token as sentence A, so it is a constant added to every position.
	TypeEmbd []float32
	EmbNorm  nn.LayerNorm
	Blocks   []BlockWeights
}

// Model is one opened checkpoint and the vocabulary in the same file.
type Model struct {
	Cfg   *Config
	W     *Weights
	Vocab *ugm.Tokenizer

	file *tensors.GGUF
	// trace, when set, is handed the waypoints llama.cpp names, for the tests
	// that locate a divergence.
	trace func(name string, rows [][]float32)

	mu      sync.Mutex // one pass at a time: the scratch is one
	scratch *scratch

	// The card, when UseVulkan was called. nomic/vulkan.go.
	dev *vk.Device
	gpu *vk.NomicPipeline
}

func Open(path string) (*Model, error) {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	m, err := New(g)
	if err != nil {
		g.Close()
		return nil, err
	}
	return m, nil
}

// New binds a GGUF the caller already opened. The model takes ownership: its
// Close closes the file. A New that fails closes nothing.
func New(g *tensors.GGUF) (*Model, error) {
	cfg, err := LoadConfig(g)
	if err != nil {
		return nil, err
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		return nil, err
	}
	vocab, err := ugm.Load(g)
	if err != nil {
		return nil, err
	}
	return &Model{Cfg: cfg, W: w, Vocab: vocab, file: g}, nil
}

func (m *Model) Close() error {
	m.closeVulkan()
	return m.file.Close()
}

// Encode is the text framed by <s> and </s> and not cut; Tokenize is the same
// cut to the context.
func (m *Model) Encode(text string) []int32 { return m.Vocab.Encode(text, true, false) }

// Context is the most positions one text may have.
func (m *Model) Context() int { return m.Cfg.Context }

// LoadWeights binds every tensor and checks each against the shape the
// configuration expects.
func LoadWeights(g *tensors.GGUF, cfg *Config) (*Weights, error) {
	w := &Weights{Blocks: make([]BlockWeights, cfg.Blocks)}
	var err error
	if w.TokenEmbd, err = matrix(g, "token_embd.weight", -1, cfg.Dim); err != nil {
		return nil, err
	}
	if w.TypeEmbd, err = floats(g, "token_types.weight", -1); err != nil {
		return nil, err
	}
	if len(w.TypeEmbd) < cfg.Dim {
		return nil, fmt.Errorf("token_types.weight holds %d floats, fewer than a row", len(w.TypeEmbd))
	}
	w.TypeEmbd = w.TypeEmbd[:cfg.Dim]
	if w.EmbNorm, err = loadNorm(g, "token_embd_norm", cfg); err != nil {
		return nil, err
	}

	for i := range w.Blocks {
		b := &w.Blocks[i]
		name := func(s string) string { return fmt.Sprintf("blk.%d.%s", i, s) }
		if b.QKV, err = biased(g, name("attn_qkv"), 3*cfg.Dim, cfg.Dim); err != nil {
			return nil, err
		}
		if b.O, err = biased(g, name("attn_output"), cfg.Dim, cfg.Dim); err != nil {
			return nil, err
		}
		if b.AttnNorm, err = loadNorm(g, name("attn_output_norm"), cfg); err != nil {
			return nil, err
		}
		if b.OutNorm, err = loadNorm(g, name("layer_output_norm"), cfg); err != nil {
			return nil, err
		}
		if !cfg.Mixture(i) {
			if b.Up, err = biased(g, name("ffn_up"), cfg.FF, cfg.Dim); err != nil {
				return nil, err
			}
			if b.Down, err = biased(g, name("ffn_down"), cfg.Dim, cfg.FF); err != nil {
				return nil, err
			}
			continue
		}
		if b.Router, err = matrix(g, name("ffn_gate_inp.weight"), cfg.Experts, cfg.Dim); err != nil {
			return nil, err
		}
		if b.UpExps, err = experts(g, name("ffn_up_exps.weight"), cfg.Experts, cfg.FF, cfg.Dim); err != nil {
			return nil, err
		}
		if b.DownExps, err = experts(g, name("ffn_down_exps.weight"), cfg.Experts, cfg.Dim, cfg.FF); err != nil {
			return nil, err
		}
	}
	return w, nil
}

// matrix binds a two-dimensional tensor. GGUF writes the row length first, so
// Shape[0] counts inputs and Shape[1] outputs. rows < 0 accepts any count.
func matrix(g *tensors.GGUF, name string, rows, cols int) (nn.Matrix, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nn.Matrix{}, fmt.Errorf("tensor %q is absent", name)
	}
	if len(t.Shape) != 2 {
		return nn.Matrix{}, fmt.Errorf("tensor %q has %d dimensions", name, len(t.Shape))
	}
	if t.Shape[0] != cols || (rows >= 0 && t.Shape[1] != rows) {
		return nn.Matrix{}, fmt.Errorf("tensor %q is %v, not [%d %d]", name, t.Shape, cols, rows)
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		return nn.Matrix{}, fmt.Errorf("tensor %q is %s, which is not a weight format", name, t.DType)
	}
	return nn.Matrix{Data: t.Raw, Quant: q, Rows: t.Shape[1], Cols: t.Shape[0]}, nil
}

// biased binds prefix.weight with prefix.bias as its bias.
func biased(g *tensors.GGUF, prefix string, rows, cols int) (nn.Matrix, error) {
	m, err := matrix(g, prefix+".weight", rows, cols)
	if err != nil {
		return m, err
	}
	if m.Bias, err = floats(g, prefix+".bias", rows); err != nil {
		return m, err
	}
	return m, nil
}

// experts cuts a stack of expert matrices, [cols rows experts] in GGUF's
// order, into one matrix per expert over the same bytes.
func experts(g *tensors.GGUF, name string, n, rows, cols int) ([]nn.Matrix, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q is absent", name)
	}
	if len(t.Shape) != 3 || t.Shape[0] != cols || t.Shape[1] != rows || t.Shape[2] != n {
		return nil, fmt.Errorf("tensor %q is %v, not [%d %d %d]", name, t.Shape, cols, rows, n)
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		return nil, fmt.Errorf("tensor %q is %s, which is not a weight format", name, t.DType)
	}
	out := make([]nn.Matrix, n)
	for e := range out {
		out[e] = nn.Matrix{Quant: q, Rows: rows, Cols: cols}
		size := out[e].RowBytes() * rows
		if len(t.Raw) != n*size {
			return nil, fmt.Errorf("tensor %q holds %d bytes, not %d experts of %d", name, len(t.Raw), n, size)
		}
		out[e].Data = t.Raw[e*size : (e+1)*size]
	}
	return out, nil
}

// floats copies an F32 tensor out of the mapping. want < 0 accepts any size.
func floats(g *tensors.GGUF, name string, want int) ([]float32, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q is absent", name)
	}
	if t.DType != "F32" {
		return nil, fmt.Errorf("tensor %q is %s, not F32", name, t.DType)
	}
	out := make([]float32, len(t.Raw)/4)
	if want >= 0 && len(out) != want {
		return nil, fmt.Errorf("tensor %q holds %d floats, not %d", name, len(out), want)
	}
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Raw[4*i:]))
	}
	return out, nil
}

func loadNorm(g *tensors.GGUF, prefix string, cfg *Config) (nn.LayerNorm, error) {
	gain, err := floats(g, prefix+".weight", cfg.Dim)
	if err != nil {
		return nn.LayerNorm{}, err
	}
	bias, err := floats(g, prefix+".bias", cfg.Dim)
	if err != nil {
		return nn.LayerNorm{}, err
	}
	return nn.LayerNorm{Gain: gain, Bias: bias, Eps: cfg.Eps}, nil
}
