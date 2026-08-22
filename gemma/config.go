// Package gemma runs Google's Gemma 4 language models from a GGUF file, in Go,
// with no cgo and nothing outside the standard library.
package gemma

// The geometry of the model, read from the file rather than assumed.
//
// Two Gemma 4 checkpoints declare the same architecture and disagree about
// almost everything measurable: E2B carries per-layer embeddings and shares its
// keys and values from block 15 on, the 12B does neither; one has eight query
// heads throughout, the other sixteen with a key-value count that changes from
// block to block. Both must run from this code, so every number below is read,
// none is written down, and a zero is an ordinary value rather than a branch.

import (
	"fmt"
	"strings"

	"github.com/ThiraSoft/golem/sample"
	"github.com/ThiraSoft/golem/tensors"
)

type BlockConfig struct {
	Index      int
	Window     bool // a sliding-window block; false means it sees everything
	WindowSize int  // positions visible, this one included; 0 when global
	Heads      int  // query heads
	KVHeads    int
	HeadDim    int
	RoPEBase   float64
	RoPEDims   int // how many elements of the head are rotated
	FFN        int
	OwnsKV     bool
	KVSource   int  // the block whose cache this one reads; Index when OwnsKV
	ValueIsKey bool // no value projection: the key projection serves as both
	MoE        bool // a router, and a second feed forward beside the dense one
}

type Config struct {
	Arch   string
	Dim    int
	Vocab  int
	Eps    float32
	PLEDim int // 0 when the model has no per-layer embeddings
	// Experts, ExpertsUsed and ExpertFFN are zero on a checkpoint whose feed
	// forward is dense. The 26B A4B declares a hundred and twenty-eight
	// experts of which eight are read per position, each seven hundred and
	// four wide, beside a shared expert that is the ordinary dense feed
	// forward and goes on using FFN.
	Experts, ExpertsUsed, ExpertFFN int
	LogitSoftcap                    float32
	MaxContext                      int
	Blocks                          []BlockConfig
	// Sampling is what the file asks to be sampled with. A checkpoint that
	// declares none of the three keys gets sample.Defaults().
	Sampling sample.Params
	// Suppress are tokens the file forbids. The 12B checkpoint can emit the
	// markers that stand for an image or a piece of audio, which no text
	// decoder can turn into anything; llama.cpp answers by adding minus
	// infinity to those logits, and so does Logits below.
	Suppress []int32
	// EmptyThought is what ChatOptions.EmptyThought means: this file's template
	// opens and closes a thought channel at the end of a generation prompt when
	// thinking is off. It is read from the Jinja rather than from the model's
	// name, because that is where the difference actually is.
	EmptyThought bool
	// ImageOpen, ImageClose and ImageSoft are the identifiers of the three
	// markers a picture is written with, or zero in a vocabulary that has
	// none. The soft one is never embedded — the row is given — but it has to
	// be a real token, because the cache and the per-layer lookup hold
	// identifiers rather than rows.
	ImageOpen, ImageClose, ImageSoft int32
}

// tokenNamed finds one piece in the vocabulary, and answers 0 when the file
// has no such token — which is a vocabulary that was never taught to carry a
// picture, not an error at load time.
func tokenNamed(g *tensors.GGUF, piece string) int32 {
	pieces, err := g.Strings("tokenizer.ggml.tokens")
	if err != nil {
		return 0
	}
	for i, p := range pieces {
		if p == piece {
			return int32(i)
		}
	}
	return 0
}

// perBlock reads a value that the format may have stored once for the whole
// model or once per block.
func perBlock(values []uint32, i int) int {
	if len(values) == 1 {
		return int(values[0])
	}
	return int(values[i])
}

// LoadConfig reads the geometry. maxContext caps the context length, which the
// file declares as 131072 — the caches for that would exhaust the machine long
// before the length became useful.
func LoadConfig(g *tensors.GGUF, maxContext int) (*Config, error) {
	arch, err := g.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "gemma4" {
		return nil, fmt.Errorf("architecture %q is not gemma4", arch)
	}
	key := func(suffix string) string { return arch + "." + suffix }

	blockCount, err := g.Uint32(key("block_count"))
	if err != nil {
		return nil, err
	}
	dim, err := g.Uint32(key("embedding_length"))
	if err != nil {
		return nil, err
	}
	eps, err := g.Float32(key("attention.layer_norm_rms_epsilon"))
	if err != nil {
		return nil, err
	}
	heads, err := g.Uint32Slice(key("attention.head_count"))
	if err != nil {
		return nil, err
	}
	kvHeads, err := g.Uint32Slice(key("attention.head_count_kv"))
	if err != nil {
		return nil, err
	}
	ffn, err := g.Uint32Slice(key("feed_forward_length"))
	if err != nil {
		return nil, err
	}
	pattern, err := g.BoolSlice(key("attention.sliding_window_pattern"))
	if err != nil {
		return nil, err
	}
	if len(pattern) != int(blockCount) {
		return nil, fmt.Errorf("the window pattern has %d entries for %d blocks", len(pattern), blockCount)
	}
	window, err := g.Uint32(key("attention.sliding_window"))
	if err != nil {
		return nil, err
	}

	headDimGlobal, err := g.Uint32(key("attention.key_length"))
	if err != nil {
		return nil, err
	}
	headDimWindow := headDimGlobal
	if v, err := g.Uint32(key("attention.key_length_swa")); err == nil {
		headDimWindow = v
	}
	ropeGlobal, err := g.Float32(key("rope.freq_base"))
	if err != nil {
		return nil, err
	}
	ropeWindow := ropeGlobal
	if v, err := g.Float32(key("rope.freq_base_swa")); err == nil {
		ropeWindow = v
	}
	ropeDimsGlobal, err := g.Uint32(key("rope.dimension_count"))
	if err != nil {
		return nil, err
	}
	ropeDimsWindow := ropeDimsGlobal
	if v, err := g.Uint32(key("rope.dimension_count_swa")); err == nil {
		ropeDimsWindow = v
	}

	cfg := &Config{
		Arch:       arch,
		Sampling:   loadSampling(g),
		Dim:        int(dim),
		Eps:        eps,
		MaxContext: maxContext,
	}
	// Optional: absent means the mechanism is not in this checkpoint.
	if v, err := g.Uint32(key("embedding_length_per_layer_input")); err == nil {
		cfg.PLEDim = int(v)
	}
	if v, err := g.Uint32(key("expert_count")); err == nil {
		cfg.Experts = int(v)
	}
	if v, err := g.Uint32(key("expert_used_count")); err == nil {
		cfg.ExpertsUsed = int(v)
	}
	if v, err := g.Uint32(key("expert_feed_forward_length")); err == nil {
		cfg.ExpertFFN = int(v)
	}
	if v, err := g.Float32(key("final_logit_softcapping")); err == nil {
		cfg.LogitSoftcap = v
	}
	if t, err := g.String("tokenizer.chat_template"); err == nil {
		cfg.EmptyThought = strings.Contains(t, emptyThoughtJinja)
	}
	cfg.ImageOpen = tokenNamed(g, imageOpen)
	cfg.ImageClose = tokenNamed(g, imageClose)
	cfg.ImageSoft = tokenNamed(g, imageSoft)
	if ids, err := g.Uint32Slice("tokenizer.ggml.suppress_tokens"); err == nil {
		cfg.Suppress = make([]int32, len(ids))
		for i, id := range ids {
			cfg.Suppress[i] = int32(id)
		}
	}

	embd, ok := g.Tensors["token_embd.weight"]
	if !ok || len(embd.Shape) != 2 {
		return nil, fmt.Errorf("token_embd.weight is absent or not two-dimensional")
	}
	cfg.Vocab = embd.Shape[1] // GGUF stores the row length first

	// Which blocks compute their own keys and values. llama.cpp reads this as a
	// count of blocks at the end that do not, so the ones that do come first.
	ownedThrough := int(blockCount)
	if shared, err := g.Uint32(key("attention.shared_kv_layers")); err == nil {
		ownedThrough = int(blockCount) - int(shared)
	}

	cfg.Blocks = make([]BlockConfig, blockCount)
	for i := range cfg.Blocks {
		b := BlockConfig{
			Index:    i,
			Window:   pattern[i],
			Heads:    perBlock(heads, i),
			KVHeads:  perBlock(kvHeads, i),
			FFN:      perBlock(ffn, i),
			OwnsKV:   i < ownedThrough,
			KVSource: i,
		}
		if b.Window {
			b.HeadDim = int(headDimWindow)
			b.RoPEBase = float64(ropeWindow)
			b.RoPEDims = int(ropeDimsWindow)
			b.WindowSize = int(window)
		} else {
			b.HeadDim = int(headDimGlobal)
			b.RoPEBase = float64(ropeGlobal)
			b.RoPEDims = int(ropeDimsGlobal)
		}
		if !b.OwnsKV {
			// The last block of the same kind that still computed a cache:
			// one back for a global block, two for a window one, because the
			// pattern puts a global block last before the sharing begins.
			back := 1
			if b.Window {
				back = 2
			}
			b.KVSource = ownedThrough - back
			if b.KVSource < 0 || b.KVSource >= ownedThrough {
				return nil, fmt.Errorf("block %d would read block %d, which owns no cache", i, b.KVSource)
			}
			if cfg.Blocks[b.KVSource].Window != b.Window {
				return nil, fmt.Errorf("block %d would read block %d, which is of the other kind", i, b.KVSource)
			}
		}
		if b.HeadDim%2 != 0 || b.RoPEDims > b.HeadDim {
			return nil, fmt.Errorf("block %d: head %d, rotated %d", i, b.HeadDim, b.RoPEDims)
		}
		if b.OwnsKV {
			// A block may publish no value projection at all, and the 12B's
			// global blocks do not. Its keys are then its values as well,
			// taken before the key norm and before the rotation. llama.cpp
			// reads the same absence the same way.
			_, hasV := g.Tensors[fmt.Sprintf("blk.%d.attn_v.weight", i)]
			b.ValueIsKey = !hasV
		}
		// A mixture block is one that carries a router. llama.cpp decides the
		// same way — models/gemma4.cpp reads is_moe_layer as ffn_gate_inp
		// being present — because the pattern of mixture blocks is in no key.
		if _, ok := g.Tensors[fmt.Sprintf("blk.%d.ffn_gate_inp.weight", i)]; ok {
			b.MoE = true
			if cfg.Experts == 0 || cfg.ExpertsUsed == 0 || cfg.ExpertFFN == 0 {
				return nil, fmt.Errorf("block %d has a router and the file declares no expert geometry", i)
			}
			if cfg.ExpertsUsed > cfg.Experts {
				return nil, fmt.Errorf("the file uses %d of %d experts", cfg.ExpertsUsed, cfg.Experts)
			}
		}
		if b.Heads%b.KVHeads != 0 {
			return nil, fmt.Errorf("block %d: %d query heads do not divide among %d key-value heads", i, b.Heads, b.KVHeads)
		}
		cfg.Blocks[i] = b
	}
	return cfg, nil
}

// loadSampling reads the generation parameters the file was published with.
// They live outside the architecture namespace, under general.sampling, and any
// of the three may be missing.
func loadSampling(g *tensors.GGUF) sample.Params {
	p := sample.Defaults()
	if v, err := g.Float32("general.sampling.temp"); err == nil {
		p.Temperature = v
	}
	if v, err := g.Uint32("general.sampling.top_k"); err == nil {
		p.TopK = int(v)
	}
	if v, err := g.Float32("general.sampling.top_p"); err == nil {
		p.TopP = v
	}
	return p
}

// MaxHeadDim is the widest head in the model, which is what scratch buffers
// have to be sized for.
func (c *Config) MaxHeadDim() int {
	max := 0
	for _, b := range c.Blocks {
		if b.HeadDim > max {
			max = b.HeadDim
		}
	}
	return max
}
