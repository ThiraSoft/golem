package gemma

import "testing"

func TestCacheAliasesSharedLayers(t *testing.T) {
	g := openModel(t)
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	c := NewCache(cfg)

	if len(c.Layers) != 35 {
		t.Fatalf("%d layers", len(c.Layers))
	}
	// The fifteen owners each have their own storage.
	seen := map[*LayerCache]int{}
	for i := 0; i < 15; i++ {
		if prev, ok := seen[c.Layers[i]]; ok {
			t.Fatalf("blocks %d and %d share storage and should not", prev, i)
		}
		seen[c.Layers[i]] = i
	}
	// The twenty sharers point at 13 or 14, and at nothing else.
	for i := 15; i < 35; i++ {
		want := 14
		if cfg.Blocks[i].Window {
			want = 13
		}
		if c.Layers[i] != c.Layers[want] {
			t.Fatalf("block %d does not read block %d's cache", i, want)
		}
	}
	// A window block keeps its window, a global one the whole context.
	if c.Layers[0].Capacity != 512 || c.Layers[4].Capacity != 4096 {
		t.Fatalf("capacities %d and %d", c.Layers[0].Capacity, c.Layers[4].Capacity)
	}
}

func TestCacheRingReplacesWhatTheWindowHides(t *testing.T) {
	lc := &LayerCache{KVHeads: 1, HeadDim: 4, Capacity: 3}
	lc.K = make([]float32, lc.Capacity*lc.KVHeads*lc.HeadDim)
	lc.V = make([]float32, len(lc.K))

	for pos := 0; pos < 5; pos++ {
		k := []float32{float32(pos), 0, 0, 0}
		v := []float32{0, float32(pos), 0, 0}
		lc.Store(pos, 0, k, v)
	}
	// Positions 2, 3 and 4 survive; 0 and 1 were overwritten by 3 and 4.
	for pos := 2; pos <= 4; pos++ {
		if got := lc.Key(pos, 0)[0]; got != float32(pos) {
			t.Fatalf("position %d holds key %v", pos, got)
		}
		if got := lc.Value(pos, 0)[1]; got != float32(pos) {
			t.Fatalf("position %d holds value %v", pos, got)
		}
	}
}

func TestCacheStoresHeadsSeparately(t *testing.T) {
	lc := &LayerCache{KVHeads: 2, HeadDim: 2, Capacity: 2}
	lc.K = make([]float32, lc.Capacity*lc.KVHeads*lc.HeadDim)
	lc.V = make([]float32, len(lc.K))

	lc.Store(0, 0, []float32{1, 2}, []float32{3, 4})
	lc.Store(0, 1, []float32{5, 6}, []float32{7, 8})

	if k := lc.Key(0, 0); k[0] != 1 || k[1] != 2 {
		t.Fatalf("head 0 key %v", k)
	}
	if k := lc.Key(0, 1); k[0] != 5 || k[1] != 6 {
		t.Fatalf("head 1 key %v", k)
	}
	if v := lc.Value(0, 1); v[0] != 7 || v[1] != 8 {
		t.Fatalf("head 1 value %v", v)
	}
}

func TestCacheVisibleRange(t *testing.T) {
	window := BlockConfig{Window: true, WindowSize: 512}
	global := BlockConfig{Window: false}
	c := &Cache{}

	if first, last := c.Visible(window, 3); first != 0 || last != 3 {
		t.Fatalf("early window: %d..%d", first, last)
	}
	// Position 600 sees 89..600: five hundred and twelve positions, itself
	// included, which is what llama.cpp's mask allows.
	if first, last := c.Visible(window, 600); first != 89 || last != 600 {
		t.Fatalf("late window: %d..%d, want 89..600", first, last)
	}
	if n := 600 - 89 + 1; n != 512 {
		t.Fatalf("the window is %d positions wide", n)
	}
	if first, last := c.Visible(global, 600); first != 0 || last != 600 {
		t.Fatalf("global: %d..%d", first, last)
	}
}
