package gemma

import (
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

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
	// A window block keeps its window and one pass more, rounded to a power of
	// two; a global one the whole context.
	if c.Layers[0].Capacity != 1024 || c.Layers[4].Capacity != 4096 {
		t.Fatalf("capacities %d and %d", c.Layers[0].Capacity, c.Layers[4].Capacity)
	}
}

// widenAll is for the failure messages: a cache holds halves, and printing the
// bit patterns says nothing to whoever has to read the test output.
func widenAll(h []uint16) []float32 {
	f := make([]float32, len(h))
	for i, v := range h {
		f[i] = nn.Widen(v)
	}
	return f
}

func TestCacheRingReplacesWhatTheWindowHides(t *testing.T) {
	lc := &LayerCache{KVHeads: 1, HeadDim: 4, Capacity: 3}
	lc.K = make([]uint16, lc.Capacity*lc.KVHeads*lc.HeadDim)
	lc.V = make([]uint16, len(lc.K))

	for pos := 0; pos < 5; pos++ {
		k := []float32{float32(pos), 0, 0, 0}
		v := []float32{0, float32(pos), 0, 0}
		lc.Store(pos, 0, k, v)
	}
	// Positions 2, 3 and 4 survive; 0 and 1 were overwritten by 3 and 4.
	for pos := 2; pos <= 4; pos++ {
		if got := nn.Widen(lc.Key(pos, 0)[0]); got != float32(pos) {
			t.Fatalf("position %d holds key %v", pos, got)
		}
		if got := nn.Widen(lc.Value(pos, 0)[1]); got != float32(pos) {
			t.Fatalf("position %d holds value %v", pos, got)
		}
	}
}

func TestCacheStoresHeadsSeparately(t *testing.T) {
	lc := &LayerCache{KVHeads: 2, HeadDim: 2, Capacity: 2}
	lc.K = make([]uint16, lc.Capacity*lc.KVHeads*lc.HeadDim)
	lc.V = make([]uint16, len(lc.K))

	lc.Store(0, 0, []float32{1, 2}, []float32{3, 4})
	lc.Store(0, 1, []float32{5, 6}, []float32{7, 8})

	if k := lc.Key(0, 0); nn.Widen(k[0]) != 1 || nn.Widen(k[1]) != 2 {
		t.Fatalf("head 0 key %v", widenAll(k))
	}
	if k := lc.Key(0, 1); nn.Widen(k[0]) != 5 || nn.Widen(k[1]) != 6 {
		t.Fatalf("head 1 key %v", widenAll(k))
	}
	if v := lc.Value(0, 1); nn.Widen(v[0]) != 7 || nn.Widen(v[1]) != 8 {
		t.Fatalf("head 1 value %v", widenAll(v))
	}
}

func TestCacheVisibleRange(t *testing.T) {
	window := BlockConfig{Window: true, WindowSize: 512}
	global := BlockConfig{Window: false}
	c := &Cache{}

	if first, last := c.Visible(window, 3, 3); first != 0 || last != 3 {
		t.Fatalf("early window: %d..%d", first, last)
	}
	// Position 600 sees 89..600: five hundred and twelve positions, itself
	// included, which is what llama.cpp's mask allows.
	if first, last := c.Visible(window, 600, 600); first != 89 || last != 600 {
		t.Fatalf("late window: %d..%d, want 89..600", first, last)
	}
	if n := 600 - 89 + 1; n != 512 {
		t.Fatalf("the window is %d positions wide", n)
	}
	if first, last := c.Visible(global, 600, 600); first != 0 || last != 600 {
		t.Fatalf("global: %d..%d", first, last)
	}

	// A token inside a picture looks forward to the end of it, and the window
	// is measured from there, so that every token of the span is given the
	// same range.
	if first, last := c.Visible(window, 500, 600); first != 89 || last != 600 {
		t.Fatalf("a span's first token: %d..%d, want 89..600", first, last)
	}
	if first, last := c.Visible(global, 500, 600); first != 0 || last != 600 {
		t.Fatalf("a global block's span: %d..%d", first, last)
	}
	// An Until behind the position is a caller's slip, not a narrower window.
	if first, last := c.Visible(global, 600, 0); first != 0 || last != 600 {
		t.Fatalf("an Until left behind: %d..%d", first, last)
	}
}

// The mask and the remainder must name the same slot, or a cache that wraps
// starts reading someone else's key.
func TestRingMaskAgreesWithTheRemainder(t *testing.T) {
	for _, capacity := range []int{1, 2, 3, 512, 1000, 1024, 4096} {
		lc := &LayerCache{KVHeads: 2, HeadDim: 4, Capacity: capacity, mask: ringMask(capacity)}
		for pos := 0; pos < capacity*3+7; pos++ {
			for head := 0; head < lc.KVHeads; head++ {
				want := ((pos%capacity)*lc.KVHeads + head) * lc.HeadDim
				if got := lc.offset(pos, head); got != want {
					t.Fatalf("capacity %d, position %d, head %d: %d against %d",
						capacity, pos, head, got, want)
				}
			}
		}
	}
}
