package qwen35

// BlockCache holds either a KV cache (for Full Attention) or an SSM state (for Gated Delta Net).
type BlockCache struct {
	// Full Attention KV cache
	Keys   []float32
	Values []float32

	// Gated Delta Net recurrent states:
	// ConvState: [ConvKernel - 1, InnerConvDim] (circular buffer for causal 1D conv)
	ConvState []float32
	// SSMState: 48 heads * 128 * 128 state matrices
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
