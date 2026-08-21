package gemma

import (
	"strconv"
	"testing"
)

func itoa(i int) string { return strconv.Itoa(i) }

// replayBlock replays one block over the whole prompt, driven by the normed
// inputs llama.cpp recorded, and returns nothing: the point is the state it
// leaves in the cache and what it wrote into the fixture comparisons.
func replayBlock(t *testing.T, f *fixture, cfg *Config, w *Weights,
	cache *Cache, s *Scratch, il int, check bool) {
	t.Helper()

	bc := cfg.Blocks[il]
	bw := &w.Blocks[il]
	freqs := w.RoPEFreqs
	if bc.Window {
		freqs = nil // the factors belong to the global blocks alone
	}
	normed := s.Batch(cfg.Dim, 1)
	out := [][]float32{make([]float32, cfg.Dim)}

	for pos := range f.Tokens {
		normed.Set(0, f.column(t, "attn_norm-"+itoa(il), pos))
		Attention(cfg, bc, bw, s.RoPE(bc, Run(cache, pos, 1), freqs), Run(cache, pos, 1), s, normed, out)

		if !check {
			continue
		}
		// kqv_out is the concatenated heads before the output projection, so
		// compare there: a mistake in the heads and a mistake in the projection
		// look the same from further downstream.
		//
		// Relative, for the reason compareRelative gives: one fp16 step in one
		// attention probability, now and then. Everything upstream of the
		// probabilities — the query, the key, the value, every score — matches
		// to within 1e-5, and TestAttentionWaypoints is where that is checked.
		compareRelative(t, "kqv_out-"+itoa(il)+" at position "+itoa(pos),
			s.Heads(bc), f.column(t, "kqv_out-"+itoa(il), pos), 1e-3)
	}
}

func TestAttentionWindowBlock(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cache, s := NewCache(cfg), NewScratch(cfg)
	replayBlock(t, f, cfg, w, cache, s, 0, true)
}

func TestAttentionGlobalBlock(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cache, s := NewCache(cfg), NewScratch(cfg)
	replayBlock(t, f, cfg, w, cache, s, 4, true)
}

// The waypoints inside one position, in order, so that a failure says which
// step went wrong rather than that the block did.
func TestAttentionWaypoints(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	s := NewScratch(cfg)

	for _, il := range []int{0, 4} {
		bc := cfg.Blocks[il]
		bw := &w.Blocks[il]
		freqs := w.RoPEFreqs
		if bc.Window {
			freqs = nil
		}
		cache := NewCache(cfg)
		out := [][]float32{make([]float32, cfg.Dim)}
		normed := s.Batch(cfg.Dim, 1)
		const pos = 3

		// Fill the cache up to the position under test.
		for p := 0; p < pos; p++ {
			normed.Set(0, f.column(t, "attn_norm-"+itoa(il), p))
			Attention(cfg, bc, bw, s.RoPE(bc, Run(cache, p, 1), freqs), Run(cache, p, 1), s, normed, out)
		}
		normed.Set(0, f.column(t, "attn_norm-"+itoa(il), pos))
		Attention(cfg, bc, bw, s.RoPE(bc, Run(cache, pos, 1), freqs), Run(cache, pos, 1), s, normed, out)

		compare(t, "Qcur_pos-"+itoa(il), s.Q(bc), f.heads(t, "Qcur_pos-"+itoa(il), pos), 2e-4)
		compare(t, "Kcur_pos-"+itoa(il), s.K(bc), f.heads(t, "Kcur_pos-"+itoa(il), pos), 2e-4)
		compare(t, "Vcur_normed-"+itoa(il), s.V(bc), f.heads(t, "Vcur_normed-"+itoa(il), pos), 2e-4)
	}
}

// A block that owns no keys reads its source's cache, and gets the same answer
// llama.cpp got. Block 15 reads block 13; block 19 reads block 14.
func TestAttentionSharedKeysAndValues(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}

	for _, pair := range []struct{ source, sharer int }{{13, 15}, {14, 19}} {
		cache, s := NewCache(cfg), NewScratch(cfg)
		// The source fills the cache; nothing is checked there, it is the
		// sharer that is on trial.
		replayBlock(t, f, cfg, w, cache, s, pair.source, false)
		replayBlock(t, f, cfg, w, cache, s, pair.sharer, true)
	}
}

// The sharer must not write. If it did, the source's keys would be silently
// replaced by keys computed from another block's input — and the model would
// go on producing fluent, wrong text.
func TestSharingBlockWritesNothing(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cache, s := NewCache(cfg), NewScratch(cfg)
	replayBlock(t, f, cfg, w, cache, s, 13, false)

	before := append([]float32(nil), cache.Layers[13].K...)
	replayBlock(t, f, cfg, w, cache, s, 15, false)

	for i := range before {
		if cache.Layers[13].K[i] != before[i] {
			t.Fatalf("block 15 modified block 13's cache at element %d", i)
		}
	}
}
