package qwen

// Where the keys and values live between tokens.
//
// Simpler than Gemma's, in the two ways that matter. Every block owns its own
// cache: there is no sharing here, so there are twenty-eight of them and no
// aliasing to be careful about. And every block sees the whole context, so a
// cache is the context long and the ring never wraps.
//
// Visible stays a method rather than becoming 0, pos at the call site. Qwen3
// dense attends to everything, but qwen35 declares a full_attention_interval,
// which makes this a real question again — and when it does, this is where the
// answer goes.
//
// The entries are held as float32 rounded through fp16, which is what
// llama.cpp's default cache type stores: keeping more precision than the
// reference would not be an improvement, it would be a divergence.

import "github.com/ThiraSoft/golem/nn"

type LayerCache struct {
	KVHeads  int
	HeadDim  int
	Capacity int
	K        []float32 // Capacity * KVHeads * HeadDim
	V        []float32

	// used is one past the highest position written since the last Reset, so
	// that Reset clears what a conversation touched rather than what the
	// context was sized for. A cache built for four thousand positions and
	// used for sixty-four is the ordinary case, and zeroing the whole of it
	// costs more than reading the prompt did.
	used int
}

func newLayerCache(kvHeads, headDim, capacity int) *LayerCache {
	n := capacity * kvHeads * headDim
	return &LayerCache{
		KVHeads:  kvHeads,
		HeadDim:  headDim,
		Capacity: capacity,
		K:        make([]float32, n),
		V:        make([]float32, n),
	}
}

// offset locates one head at one position.
func (c *LayerCache) offset(pos, head int) int {
	return ((pos%c.Capacity)*c.KVHeads + head) * c.HeadDim
}

func (c *LayerCache) Store(pos, head int, k, v []float32) {
	if pos >= c.used {
		c.used = pos + 1
	}
	o := c.offset(pos, head)
	for i := 0; i < c.HeadDim; i++ {
		c.K[o+i] = nn.RoundHalf(k[i])
		c.V[o+i] = nn.RoundHalf(v[i])
	}
}

func (c *LayerCache) Key(pos, head int) []float32 {
	o := c.offset(pos, head)
	return c.K[o : o+c.HeadDim]
}

func (c *LayerCache) Value(pos, head int) []float32 {
	o := c.offset(pos, head)
	return c.V[o : o+c.HeadDim]
}

// Cache holds one entry per block.
type Cache struct {
	Layers []*LayerCache
}

func NewCache(cfg *Config) *Cache {
	c := &Cache{Layers: make([]*LayerCache, len(cfg.Blocks))}
	for i, b := range cfg.Blocks {
		c.Layers[i] = newLayerCache(b.KVHeads, b.HeadDim, cfg.MaxContext)
	}
	return c
}

// Visible gives the inclusive range of positions a block may attend to from
// pos. Every block here sees everything before it, so the range is the whole
// prefix; the signature takes a BlockConfig because the next architecture's
// answer will depend on it.
func (c *Cache) Visible(b BlockConfig, pos int) (first, last int) {
	return 0, pos
}

// Reset forgets the conversation without releasing the memory.
//
// Only the positions that were written are cleared. Nothing beyond them can be
// read — Visible stops at the query's own position, and a position is always
// stored before it is attended to — but clearing them is what makes that an
// invariant of the cache rather than a property of the caller.
func (c *Cache) Reset() {
	for _, lc := range c.Layers {
		if lc == nil {
			continue
		}
		n := lc.used * lc.KVHeads * lc.HeadDim
		if n > len(lc.K) {
			n = len(lc.K)
		}
		clear(lc.K[:n])
		clear(lc.V[:n])
		lc.used = 0
	}
}
