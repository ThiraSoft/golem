package gemma

// Where the keys and values live between tokens.
//
// Two decisions are worth stating. A sliding-window block is given a ring of
// exactly its window, because a position that has fallen out of the ring is a
// position the mask would have hidden anyway — the storage and the rule agree,
// so neither has to be checked against the other. And a block that shares its
// keys and values is given the *same* LayerCache pointer as the block it reads
// from, not a copy: there is one cache and twenty-one blocks reading it, which
// is what makes E2B small.
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

// offset locates one head at one position. The ring wraps for window blocks;
// for global ones Capacity is the whole context and it never does.
func (c *LayerCache) offset(pos, head int) int {
	return ((pos%c.Capacity)*c.KVHeads + head) * c.HeadDim
}

func (c *LayerCache) Store(pos, head int, k, v []float32) {
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

// Cache holds one entry per block. Blocks that share point at the same one.
type Cache struct {
	Layers []*LayerCache
}

func NewCache(cfg *Config) *Cache {
	c := &Cache{Layers: make([]*LayerCache, len(cfg.Blocks))}
	for i, b := range cfg.Blocks {
		if !b.OwnsKV {
			c.Layers[i] = c.Layers[b.KVSource]
			continue
		}
		capacity := cfg.MaxContext
		if b.Window && b.WindowSize < capacity {
			capacity = b.WindowSize
		}
		c.Layers[i] = newLayerCache(b.KVHeads, b.HeadDim, capacity)
	}
	return c
}

// Visible gives the inclusive range of positions a block may attend to from
// pos. llama.cpp masks a key position p0 when p1 - p0 >= the window, so a
// window of 512 leaves 512 positions counting the query's own.
func (c *Cache) Visible(b BlockConfig, pos int) (first, last int) {
	first = 0
	if b.Window && pos-b.WindowSize+1 > 0 {
		first = pos - b.WindowSize + 1
	}
	return first, pos
}

// Reset forgets everything without releasing the memory. Distinct pointers
// only: a shared cache would otherwise be cleared as many times as it is read.
func (c *Cache) Reset() {
	done := map[*LayerCache]bool{}
	for _, lc := range c.Layers {
		if lc == nil || done[lc] {
			continue
		}
		done[lc] = true
		for i := range lc.K {
			lc.K[i] = 0
			lc.V[i] = 0
		}
	}
}
