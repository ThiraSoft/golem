// Package laya runs Laya, the open decision model Convai Innovations published
// as convaiinnovations/laya: a state and typed questions in, a calibrated
// probability for every option out, in one pass and without generating.
//
// It is a ModernBERT encoder with a small head on top. Each question is its
// own sequence, [CLS] question [SEP] [MASK] option [MASK] option … [SEP] state
// [SEP], and what the head writes at an option's [MASK] is scored into that
// option's logit. The logits of one question are divided by a temperature
// fitted after training and put through a softmax.
//
// The encoder is ModernBERT-large: twenty-eight pre-norm blocks, no biases, a
// norm without a bias, a gated GELU feed forward, and a NeoX rotation whose
// base depends on the block. Every third block, from the first, sees the whole
// sequence; the others see sixty-four positions either side. The head is two
// of PyTorch's own nn.TransformerEncoderLayer, pre-norm, with biases and a
// ReLU, over every position, after a row saying which kind of question it is
// has been added to each.
//
// The reference is the checkpoint's own rl_common.py and rl_agent_api.py, run
// by PyTorch on the processor in float32; ref/laya/dump.py records it.
package laya

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bpe"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// Question kinds, numbered as the checkpoint's type embedding is.
const (
	Choice = 0
	Score  = 1
	Noul   = 2
)

var kindNames = [3]string{"choice", "score", "noul"}

type Config struct {
	Dim, Heads, HeadDim, FF, Blocks int
	// GlobalEvery says which blocks see the whole sequence: block i does when
	// i%GlobalEvery is zero. The others see Window positions either side.
	GlobalEvery, Window int
	GlobalBase          float64
	LocalBase           float64
	Eps                 float32
	HeadLayers          int
	HeadFF              int
	MaxLen              int
	HeadMaxLen          int
	CLS, SEP, Mask      int32
	// MaskText is how the mask token is written, which build_sequence takes
	// out of every text before encoding it.
	MaskText    string
	Temperature [3]float32
	// TemperatureBy is the temperature per kind and number of options, keyed
	// as temp_bucket keys it: "choice:3-5", "noul:2".
	TemperatureBy map[string]float32
}

// Global says whether block i attends to the whole sequence.
func (c *Config) Global(i int) bool { return i%c.GlobalEvery == 0 }

type EncoderBlock struct {
	AttnNorm nn.LayerNorm // the first block has none: Gain is nil
	QKV, O   nn.Matrix
	MLPNorm  nn.LayerNorm
	Up, Down nn.Matrix // Up is 2·FF rows, the input half then the gate half
}

type HeadBlock struct {
	Norm1, Norm2 nn.LayerNorm
	QKV, O       nn.Matrix
	Up, Down     nn.Matrix
}

type Weights struct {
	TokenEmbd nn.Matrix
	EmbNorm   nn.LayerNorm
	Encoder   []EncoderBlock
	FinalNorm nn.LayerNorm
	TypeEmbd  [3][]float32
	Head      []HeadBlock
	ScoreNorm nn.LayerNorm
	ScoreHid  nn.Matrix
	ScoreOut  nn.Matrix // one row
	ActHid    nn.Matrix
	ActOut    nn.Matrix
}

// Model is one opened checkpoint directory.
type Model struct {
	Cfg   *Config
	W     *Weights
	Vocab Vocab

	file *tensors.Model

	// trace, when set, is handed the waypoints ref/laya/dump.py records.
	trace func(name string, rows [][]float32)

	// turn holds one token, taken for a pass.
	turn chan struct{}

	gpu *gpu

	// actW is the act head's first matrix widened to float32, one row per
	// output: it is Dim+4 wide, which no fp16 kernel takes.
	actW []float32

	// packs are the matrices laid out for the processor's blocked product,
	// keyed by where their weights start; made on the first pass the
	// processor runs.
	packs    map[*byte]nn.PackedF16
	packOnce sync.Once
}

// Open reads the checkpoint as Hugging Face ships it: model.safetensors,
// rl_agent_config.json, encoder/config.json and tokenizer/tokenizer.json.
func Open(dir string) (*Model, error) {
	cfg, err := loadConfig(dir)
	if err != nil {
		return nil, err
	}
	vocab, err := loadVocab(filepath.Join(dir, "tokenizer"))
	if err != nil {
		return nil, err
	}
	var names struct {
		CLS  string `json:"cls_token"`
		SEP  string `json:"sep_token"`
		Mask string `json:"mask_token"`
	}
	if err := readJSON(filepath.Join(dir, "tokenizer", "tokenizer_config.json"), &names); err != nil {
		return nil, err
	}
	cfg.MaskText = names.Mask
	for _, c := range []struct {
		name string
		dst  *int32
	}{{names.CLS, &cfg.CLS}, {names.SEP, &cfg.SEP}, {names.Mask, &cfg.Mask}} {
		id, ok := vocab.ID(c.name)
		if c.name == "" || !ok {
			return nil, fmt.Errorf("laya: the tokenizer has no %q", c.name)
		}
		*c.dst = id
	}
	f, err := tensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	w, err := loadWeights(f, cfg)
	if err != nil {
		f.Close()
		return nil, err
	}
	actW := make([]float32, w.ActHid.Rows*w.ActHid.Cols)
	for r := 0; r < w.ActHid.Rows; r++ {
		w.ActHid.Row(r, actW[r*w.ActHid.Cols:(r+1)*w.ActHid.Cols])
	}
	return &Model{Cfg: cfg, W: w, Vocab: vocab, file: f, turn: make(chan struct{}, 1), actW: actW}, nil
}

// Vocab is what the model needs of a tokenizer. The English checkpoints carry
// ModernBERT's byte-level BPE, the multilingual one mmBERT's SentencePiece
// kind; tokenizer.json says which.
type Vocab interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	ID(text string) (int32, bool)
}

func loadVocab(dir string) (Vocab, error) {
	path := filepath.Join(dir, "tokenizer.json")
	var head struct {
		Pre struct {
			Type string `json:"type"`
		} `json:"pre_tokenizer"`
	}
	if err := readJSON(path, &head); err != nil {
		return nil, err
	}
	switch head.Pre.Type {
	case "ByteLevel":
		return bytebpe.LoadTokenizerJSON(path)
	case "Metaspace":
		return bpe.LoadTokenizerJSON(path)
	}
	return nil, fmt.Errorf("laya: pre-tokenizer %q is not one golem reads", head.Pre.Type)
}

func (m *Model) Close() error {
	m.closeVulkan()
	return m.file.Close()
}

func loadConfig(dir string) (*Config, error) {
	var enc struct {
		Hidden         int     `json:"hidden_size"`
		Heads          int     `json:"num_attention_heads"`
		Layers         int     `json:"num_hidden_layers"`
		FF             int     `json:"intermediate_size"`
		GlobalEvery    int     `json:"global_attn_every_n_layers"`
		Local          int     `json:"local_attention"`
		Eps            float32 `json:"norm_eps"`
		NormBias       bool    `json:"norm_bias"`
		AttnBias       bool    `json:"attention_bias"`
		MLPBias        bool    `json:"mlp_bias"`
		Activation     string  `json:"hidden_activation"`
		GlobalRoPEBase float64 `json:"global_rope_theta"`
		LocalRoPEBase  float64 `json:"local_rope_theta"`
		RoPE           map[string]struct {
			Theta float64 `json:"rope_theta"`
			Type  string  `json:"rope_type"`
		} `json:"rope_parameters"`
	}
	if err := readJSON(filepath.Join(dir, "encoder", "config.json"), &enc); err != nil {
		return nil, err
	}
	var agent struct {
		HeadLayers    int                `json:"head_layers"`
		MaxLen        int                `json:"max_len"`
		HeadMaxLen    int                `json:"head_max_len"`
		Temperature   []float32          `json:"temperature"`
		TemperatureBy map[string]float32 `json:"temperature_by_options"`
	}
	if err := readJSON(filepath.Join(dir, "rl_agent_config.json"), &agent); err != nil {
		return nil, err
	}
	if enc.NormBias || enc.AttnBias || enc.MLPBias {
		return nil, fmt.Errorf("laya: the encoder declares biases, which ModernBERT-large has none of")
	}
	if enc.Activation != "gelu" {
		return nil, fmt.Errorf("laya: the encoder's activation %q is not gelu", enc.Activation)
	}
	c := &Config{
		Dim: enc.Hidden, Heads: enc.Heads, FF: enc.FF, Blocks: enc.Layers,
		GlobalEvery: enc.GlobalEvery, Window: enc.Local / 2, Eps: enc.Eps,
		GlobalBase: enc.GlobalRoPEBase, LocalBase: enc.LocalRoPEBase,
		HeadLayers: agent.HeadLayers, MaxLen: agent.MaxLen, HeadMaxLen: agent.HeadMaxLen,
		Temperature: [3]float32{1, 1, 1}, TemperatureBy: agent.TemperatureBy,
	}
	// transformers 5 writes the bases per kind of block; older files write
	// them flat.
	if p, ok := enc.RoPE["full_attention"]; ok {
		c.GlobalBase = p.Theta
	}
	if p, ok := enc.RoPE["sliding_attention"]; ok {
		c.LocalBase = p.Theta
	}
	for _, p := range enc.RoPE {
		if p.Type != "" && p.Type != "default" {
			return nil, fmt.Errorf("laya: rope type %q is not implemented", p.Type)
		}
	}
	copy(c.Temperature[:], agent.Temperature)
	if c.Heads == 0 || c.Dim%c.Heads != 0 || c.GlobalEvery == 0 || c.GlobalBase == 0 || c.LocalBase == 0 {
		return nil, fmt.Errorf("laya: encoder/config.json is missing a dimension")
	}
	if c.Eps == 0 {
		c.Eps = 1e-5
	}
	c.HeadDim = c.Dim / c.Heads
	c.HeadFF = 4 * c.Dim
	return c, nil
}

func loadWeights(f *tensors.Model, cfg *Config) (*Weights, error) {
	d := cfg.Dim
	w := &Weights{Encoder: make([]EncoderBlock, cfg.Blocks), Head: make([]HeadBlock, cfg.HeadLayers)}
	var err error
	must := func(m nn.Matrix, e error) nn.Matrix {
		if err == nil {
			err = e
		}
		return m
	}
	norm := func(prefix string, bias bool) nn.LayerNorm {
		n := nn.LayerNorm{Eps: cfg.Eps}
		if err != nil {
			return n
		}
		if n.Gain, err = floats(f, prefix+".weight", d); err != nil {
			return n
		}
		if bias {
			n.Bias, err = floats(f, prefix+".bias", d)
		} else {
			n.Bias = make([]float32, d)
		}
		return n
	}

	w.TokenEmbd = must(matrix(f, "encoder.embeddings.tok_embeddings.weight", -1, d, false))
	w.EmbNorm = norm("encoder.embeddings.norm", false)
	for i := range w.Encoder {
		b := &w.Encoder[i]
		p := fmt.Sprintf("encoder.layers.%d.", i)
		if i > 0 {
			b.AttnNorm = norm(p+"attn_norm", false)
		}
		b.QKV = must(matrix(f, p+"attn.Wqkv.weight", 3*d, d, false))
		b.O = must(matrix(f, p+"attn.Wo.weight", d, d, false))
		b.MLPNorm = norm(p+"mlp_norm", false)
		b.Up = must(matrix(f, p+"mlp.Wi.weight", 2*cfg.FF, d, false))
		b.Down = must(matrix(f, p+"mlp.Wo.weight", d, cfg.FF, false))
	}
	w.FinalNorm = norm("encoder.final_norm", false)

	if err == nil {
		var types []float32
		if types, err = floats(f, "type_emb.weight", 3*d); err == nil {
			for k := range w.TypeEmbd {
				w.TypeEmbd[k] = types[k*d : (k+1)*d]
			}
		}
	}
	for i := range w.Head {
		b := &w.Head[i]
		p := fmt.Sprintf("head.layers.%d.", i)
		b.Norm1 = norm(p+"norm1", true)
		b.Norm2 = norm(p+"norm2", true)
		b.QKV = must(matrix(f, p+"self_attn.in_proj_weight", 3*d, d, false))
		if err == nil {
			b.QKV.Bias, err = floats(f, p+"self_attn.in_proj_bias", 3*d)
		}
		b.O = must(matrix(f, p+"self_attn.out_proj", d, d, true))
		b.Up = must(matrix(f, p+"linear1", cfg.HeadFF, d, true))
		b.Down = must(matrix(f, p+"linear2", d, cfg.HeadFF, true))
	}
	w.ScoreNorm = norm("scorer.0", true)
	w.ScoreHid = must(matrix(f, "scorer.1", d, d, true))
	w.ScoreOut = must(matrix(f, "scorer.3", 1, d, true))
	w.ActHid = must(matrix(f, "act_head.0", -1, d+4, true))
	if err == nil {
		w.ActOut = must(matrix(f, "act_head.2", -1, w.ActHid.Rows, true))
	}
	if err != nil {
		return nil, err
	}
	return w, nil
}

// matrix binds a two-dimensional tensor, [rows cols] as safetensors writes
// it, which is one row per output as the products read it. With biased, name
// is a module's and its .weight and .bias are both bound. rows < 0 accepts any
// count.
func matrix(f *tensors.Model, name string, rows, cols int, biased bool) (nn.Matrix, error) {
	wname := name
	if biased {
		wname = name + ".weight"
	}
	t, err := f.Get(wname)
	if err != nil {
		return nn.Matrix{}, err
	}
	if len(t.Shape) != 2 || t.Shape[1] != cols || (rows >= 0 && t.Shape[0] != rows) {
		return nn.Matrix{}, fmt.Errorf("laya: %s is %v, not [%d %d]", wname, t.Shape, rows, cols)
	}
	m := nn.Matrix{Data: t.Raw, Rows: t.Shape[0], Cols: cols}
	switch t.DType {
	case "F16":
		m.Quant = nn.F16
	case "BF16":
		m.Quant = nn.BF16
	case "F32":
		m.Quant = nn.F32
	default:
		return nn.Matrix{}, fmt.Errorf("laya: %s is %s, which is not a weight format", wname, t.DType)
	}
	if biased {
		if m.Bias, err = floats(f, name+".bias", m.Rows); err != nil {
			return nn.Matrix{}, err
		}
	}
	return m, nil
}

// floats reads a tensor of any float format into float32. want < 0 accepts
// any size.
func floats(f *tensors.Model, name string, want int) ([]float32, error) {
	t, err := f.Get(name)
	if err != nil {
		return nil, err
	}
	var out []float32
	switch t.DType {
	case "F16":
		out = make([]float32, len(t.Raw)/2)
		for i := range out {
			out[i] = nn.HalfToFloat(binary.LittleEndian.Uint16(t.Raw[2*i:]))
		}
	default:
		if out, err = t.F32(); err != nil {
			return nil, fmt.Errorf("laya: %s: %w", name, err)
		}
	}
	if want >= 0 && len(out) != want {
		return nil, fmt.Errorf("laya: %s holds %d floats, not %d", name, len(out), want)
	}
	return out, nil
}

func readJSON(path string, into any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
