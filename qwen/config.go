// Package qwen runs Qwen3 dense language models from a GGUF file, in Go, with
// no cgo and nothing outside the standard library.
package qwen

// The geometry of the model, read from the file rather than assumed.
//
// This checkpoint's twenty-eight blocks are alike in every measurable way, and
// that is a fact about the file, not about the architecture — qwen35 declares a
// full_attention_interval and will not be alike. So the geometry stays per
// block here, and the sameness is something the tests assert rather than
// something this code relies on.
//
// What Gemma's configuration carries and this one does not is most of it: no
// per-layer embeddings, no logit softcap, no sliding window, no blocks reading
// another block's cache, and no value projection standing in for a key. Their
// absence is not a zero to branch on — the fields are simply not here.

import (
	"fmt"

	"github.com/ThiraSoft/golem/sample"
	"github.com/ThiraSoft/golem/tensors"
)

type BlockConfig struct {
	Index    int
	Heads    int // query heads
	KVHeads  int
	HeadDim  int
	RoPEBase float64
	RoPEDims int // how many elements of the head are rotated
	FFN      int
}

type Config struct {
	Arch       string
	Dim        int
	Vocab      int
	Eps        float32
	MaxContext int
	Blocks     []BlockConfig
	// Sampling is what the file asks to be sampled with. A checkpoint that
	// declares none of the three keys gets sample.Defaults().
	Sampling sample.Params
}

// perBlock reads a value that the format may have stored once for the whole
// model or once per block. This file stores each of them once; the helper is
// what makes that a fact read from the file rather than an assumption.
func perBlock(values []uint32, i int) int {
	if len(values) == 1 {
		return int(values[0])
	}
	return int(values[i])
}

// LoadConfig reads the geometry. maxContext caps the context length, which the
// file declares as 40960 — the caches for that would exhaust the machine long
// before the length became useful.
func LoadConfig(g *tensors.GGUF, maxContext int) (*Config, error) {
	arch, err := g.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != "qwen3" {
		return nil, fmt.Errorf("architecture %q is not qwen3", arch)
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
	headDim, err := g.Uint32(key("attention.key_length"))
	if err != nil {
		return nil, err
	}
	// The value length is declared separately and is the same here. A file
	// where they differ would need a cache of two widths, which this does not
	// have — so it is refused rather than half-supported.
	if valueDim, err := g.Uint32(key("attention.value_length")); err == nil && valueDim != headDim {
		return nil, fmt.Errorf("keys are %d wide and values %d, which this engine does not do", headDim, valueDim)
	}
	ropeBase, err := g.Float32(key("rope.freq_base"))
	if err != nil {
		return nil, err
	}
	// How much of each head rotates. The 0.6B declares nothing and rotates all
	// of it; the 27B declares a count, so it is read when it is there.
	ropeDims := headDim
	if v, err := g.Uint32(key("rope.dimension_count")); err == nil {
		ropeDims = v
	}

	cfg := &Config{
		Arch:       arch,
		Sampling:   loadSampling(g),
		Dim:        int(dim),
		Eps:        eps,
		MaxContext: maxContext,
	}

	// The vocabulary is the embedding's own shape, not a declared number: the
	// head reads the same matrix the other way round, and a mismatch between
	// the two would be a silent walk off the end of a row.
	embd, ok := g.Tensors["token_embd.weight"]
	if !ok || len(embd.Shape) != 2 {
		return nil, fmt.Errorf("token_embd.weight is absent or not two-dimensional")
	}
	cfg.Vocab = embd.Shape[1] // GGUF stores the row length first

	cfg.Blocks = make([]BlockConfig, blockCount)
	for i := range cfg.Blocks {
		b := BlockConfig{
			Index:    i,
			Heads:    perBlock(heads, i),
			KVHeads:  perBlock(kvHeads, i),
			FFN:      perBlock(ffn, i),
			HeadDim:  int(headDim),
			RoPEBase: float64(ropeBase),
			RoPEDims: int(ropeDims),
		}
		if b.HeadDim%2 != 0 || b.RoPEDims > b.HeadDim {
			return nil, fmt.Errorf("block %d: head %d, rotated %d", i, b.HeadDim, b.RoPEDims)
		}
		if b.KVHeads == 0 || b.Heads%b.KVHeads != 0 {
			return nil, fmt.Errorf("block %d: %d query heads do not divide into %d key-value heads",
				i, b.Heads, b.KVHeads)
		}
		cfg.Blocks[i] = b
	}
	return cfg, nil
}

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
