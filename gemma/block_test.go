package gemma

import "testing"

// blockInput is what the reference fed the block: the scaled embedding for
// block 0, the previous block's output for every other.
func blockInput(t *testing.T, f *fixture, il, pos int) []float32 {
	t.Helper()
	if il == 0 {
		return f.column(t, "inp_scaled", pos)
	}
	return f.column(t, "l_out-"+itoa(il-1), pos)
}

// replayFullBlock runs one block over the whole prompt from the reference's own
// inputs, and compares its output at every position.
func replayFullBlock(t *testing.T, f *fixture, cfg *Config, w *Weights,
	cache *Cache, s *Scratch, il int, check bool) {
	t.Helper()

	bc := cfg.Blocks[il]
	bw := &w.Blocks[il]
	freqs := w.RoPEFreqs
	if bc.Window {
		freqs = nil
	}

	xs := [][]float32{make([]float32, cfg.Dim)}
	ple := [][]float32{make([]float32, len(cfg.Blocks)*cfg.PLEDim)}
	embedded := s.Batch(cfg.Dim, 1)

	for pos, token := range f.Tokens {
		copy(xs[0], blockInput(t, f, il, pos))

		embedded.Set(0, f.column(t, "inp_scaled", pos))
		PerLayerInputs(cfg, w, s, embedded, []int32{token}, ple)

		blockPLE := [][]float32{ple[0][il*cfg.PLEDim : (il+1)*cfg.PLEDim]}
		Block(cfg, bc, bw, s.RoPE(bc, Run(cache, pos, 1), freqs), Run(cache, pos, 1), s, xs, blockPLE)

		if check {
			compareRelative(t, "l_out-"+itoa(il)+" at position "+itoa(pos),
				xs[0], f.column(t, "l_out-"+itoa(il), pos), 2e-3)
		}
	}
}

func TestBlockWindowAndGlobal(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, il := range []int{0, 4} {
		cache, s := NewCache(cfg), NewScratch(cfg)
		replayFullBlock(t, f, cfg, w, cache, s, il, true)
	}
}

// The first block that owns nothing. Its source runs first, to fill the cache
// it will read; only the sharer is on trial.
func TestBlockSharingKeysAndValues(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, pair := range []struct{ source, sharer int }{{13, 15}, {14, 19}} {
		cache, s := NewCache(cfg), NewScratch(cfg)
		replayFullBlock(t, f, cfg, w, cache, s, pair.source, true)
		replayFullBlock(t, f, cfg, w, cache, s, pair.sharer, true)
	}
}

// The waypoints inside the block, so that a failure names the half it happened
// in rather than the block.
func TestBlockWaypoints(t *testing.T) {
	f := loadFixture(t, "layers")
	g := openModel(t)
	cfg, _ := LoadConfig(g, 4096)
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	const il, pos = 0, 3
	bc, bw := cfg.Blocks[il], &w.Blocks[il]
	cache, s := NewCache(cfg), NewScratch(cfg)

	xs := [][]float32{make([]float32, cfg.Dim)}
	ple := [][]float32{make([]float32, len(cfg.Blocks)*cfg.PLEDim)}
	embedded := s.Batch(cfg.Dim, 1)

	for p := 0; p <= pos; p++ {
		copy(xs[0], blockInput(t, f, il, p))
		embedded.Set(0, f.column(t, "inp_scaled", p))
		PerLayerInputs(cfg, w, s, embedded, []int32{f.Tokens[p]}, ple)
		blockPLE := [][]float32{ple[0][il*cfg.PLEDim : (il+1)*cfg.PLEDim]}
		Block(cfg, bc, bw, s.RoPE(bc, Run(cache, p, 1), nil), Run(cache, p, 1), s, xs, blockPLE)
	}

	compareRelative(t, "attn_out-"+itoa(il), s.AttnOut(), f.column(t, "attn_out-"+itoa(il), pos), 1e-3)
	compareRelative(t, "pe_in-"+itoa(il), s.PEIn(), f.column(t, "pe_in-"+itoa(il), pos), 1e-3)
	compareRelative(t, "l_out-"+itoa(il), xs[0], f.column(t, "l_out-"+itoa(il), pos), 2e-3)
}
