package qwen

import "testing"

func tinyConfig() *Config {
	return &Config{
		Dim:        8,
		Eps:        1e-6,
		MaxContext: 8,
		Blocks: []BlockConfig{
			{Index: 0, Heads: 2, KVHeads: 2, HeadDim: 4, RoPEBase: 1e6, RoPEDims: 4, FFN: 16},
			{Index: 1, Heads: 2, KVHeads: 2, HeadDim: 4, RoPEBase: 1e6, RoPEDims: 4, FFN: 16},
		},
	}
}

// Every block sees everything before it, the query's own position included.
func TestCacheSeesTheWholePrefix(t *testing.T) {
	cfg := tinyConfig()
	c := NewCache(cfg)

	first, last := c.Visible(cfg.Blocks[0], 5)
	if first != 0 || last != 5 {
		t.Fatalf("visible [%d, %d], want [0, 5]", first, last)
	}
	if first, last := c.Visible(cfg.Blocks[0], 0); first != 0 || last != 0 {
		t.Fatalf("at position zero, visible [%d, %d], want [0, 0]", first, last)
	}
}

// Nothing is shared here: twenty-eight blocks, twenty-eight caches. A stray
// alias would make one block's keys another's and never raise anything.
func TestEveryBlockOwnsItsCache(t *testing.T) {
	cfg := tinyConfig()
	c := NewCache(cfg)

	if c.Layers[0] == c.Layers[1] {
		t.Fatal("two blocks share one cache")
	}
	c.Layers[0].Store(0, 0, []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8})
	if got := c.Layers[1].Key(0, 0); got[0] != 0 {
		t.Errorf("writing block 0's cache reached block 1: %v", got)
	}
}

// What goes in comes back, at the head and position it was written to — and
// rounded through fp16, because that is what llama.cpp's cache stores.
func TestCacheStoresAndReadsBack(t *testing.T) {
	c := NewCache(tinyConfig())

	k := []float32{1, 2, 3, 4}
	v := []float32{5, 6, 7, 8}
	c.Layers[0].Store(3, 1, k, v)

	if got := c.Layers[0].Key(3, 1); got[0] != 1 || got[3] != 4 {
		t.Errorf("key came back as %v", got)
	}
	if got := c.Layers[0].Value(3, 1); got[0] != 5 || got[3] != 8 {
		t.Errorf("value came back as %v", got)
	}
	// A different head at the same position is untouched.
	if got := c.Layers[0].Key(3, 0); got[0] != 0 {
		t.Errorf("head 0 was written too: %v", got)
	}

	c.Reset()
	if got := c.Layers[0].Key(3, 1); got[0] != 0 {
		t.Error("Reset did not clear the cache")
	}
}

// The value stored is the value fp16 can hold, not the one handed in.
func TestCacheRoundsThroughHalf(t *testing.T) {
	c := NewCache(tinyConfig())

	// 1/3 is not representable in fp16 and must come back changed. Store reads
	// a whole head, so a whole head is what it is given.
	third := []float32{1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3}
	c.Layers[0].Store(0, 0, third, third)
	got := c.Layers[0].Key(0, 0)[0]
	if got == float32(1.0/3) {
		t.Error("the key was stored at full precision, not through fp16")
	}
	if got < 0.33 || got > 0.34 {
		t.Errorf("the key came back as %g, which is not 1/3 rounded", got)
	}
}

// Scratch sizes itself from the widest block, and grows once per batch width.
func TestScratchReserveGrowsOnce(t *testing.T) {
	cfg := tinyConfig()
	cfg.Blocks[1].FFN = 64 // a wider block, to prove the maximum is taken
	s := NewScratch(cfg)

	if len(s.up[0]) != 64 {
		t.Errorf("the feed-forward buffer is %d wide, want the widest block's 64", len(s.up[0]))
	}
	s.Reserve(4)
	if len(s.q) != 4 {
		t.Fatalf("after Reserve(4) there are %d rows", len(s.q))
	}
	before := &s.q[0][0]
	s.Reserve(2) // smaller: nothing should move
	if &s.q[0][0] != before {
		t.Error("Reserve reallocated for a smaller batch")
	}
}
