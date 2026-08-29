// Package qwen35 runs Qwen3.8 from a GGUF file, in Go, with no cgo. The file
// declares general.architecture = "qwen35" — llama.cpp's own models/qwen35.cpp
// does the same — and the package is named for what the file says rather than
// for what the model is called.
//
// What separates it from qwen: three blocks in four are gated delta nets, which
// keep a recurrent state rather than a ring of keys, and the checkpoint carries
// a sixty-fifth block that drafts the token after next. See mtp.go and
// speculate.go for that half.
package qwen35

// BlockCache holds either a KV cache (a full-attention block) or the recurrent
// state of a gated delta net.
type BlockCache struct {
	// Full Attention KV cache
	Keys   []float32
	Values []float32

	// Gated Delta Net recurrent states.
	//
	// ConvState is the causal convolution's window: SSMConvKernel-1 positions
	// of the convolution's own width, which is the queries, the keys and the
	// values laid end to end — SSMGroupCount*SSMStateSize*2 + SSMInnerSize.
	ConvState []float32
	// SSMState is one state matrix per head: SSMTimeStepRank of them, each
	// SSMStateSize by SSMStateSize — 48 of 128 by 128 on the 27B.
	SSMState []float32
}

type Cache struct {
	Blocks []BlockCache
}

func NewCache(cfg *Config) *Cache {
	c := &Cache{
		Blocks: make([]BlockCache, len(cfg.Blocks)),
	}
	for i, bc := range cfg.Blocks {
		newBlockCacheInto(&c.Blocks[i], cfg, bc)
	}
	return c
}

func newBlockCache(cfg *Config, bc BlockConfig) *BlockCache {
	b := &BlockCache{}
	return newBlockCacheInto(b, cfg, bc)
}

func newBlockCacheInto(b *BlockCache, cfg *Config, bc BlockConfig) *BlockCache {
	if bc.Type == BlockFullAttn {
		b.Keys = make([]float32, cfg.MaxContext*bc.KVHeads*bc.HeadDim)
		b.Values = make([]float32, cfg.MaxContext*bc.KVHeads*bc.HeadDim)
		return b
	}
	convDim := bc.SSMGroupCount*bc.SSMStateSize*2 + bc.SSMInnerSize
	b.ConvState = make([]float32, (bc.SSMConvKernel-1)*convDim)
	b.SSMState = make([]float32, bc.SSMTimeStepRank*bc.SSMStateSize*bc.SSMStateSize)
	return b
}

func (c *Cache) Reset() {
	for i := range c.Blocks {
		b := &c.Blocks[i]
		clear(b.Keys)
		clear(b.Values)
		clear(b.ConvState)
		clear(b.SSMState)
	}
}
