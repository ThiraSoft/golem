package qwen35

import "testing"

// Four consecutive slots are a square, and the whole grid is covered once.
func TestPatchOrderIsBlockMajor(t *testing.T) {
	cfg := &VisionConfig{Merge: 2}
	order := cfg.PatchOrder(4, 4)
	if len(order) != 16 {
		t.Fatalf("%d slots, wanted 16", len(order))
	}
	if got := order[:4]; got[0] != 0 || got[1] != 1 || got[2] != 4 || got[3] != 5 {
		t.Errorf("first block %v, wanted [0 1 4 5]", got)
	}
	seen := make([]bool, 16)
	for _, p := range order {
		if seen[p] {
			t.Fatalf("patch %d appears twice", p)
		}
		seen[p] = true
	}
}

// A slot's position is its patch's row and column, not its index — and it
// walks the loop clip.cpp fills its own position tensor from.
func TestPatchPositionsFollowTheGrid(t *testing.T) {
	cfg := &VisionConfig{Merge: 2}
	pos := cfg.PatchPositions(4, 4)
	for i, want := range [][2]int{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 2}} {
		if pos[i] != want {
			t.Errorf("slot %d: %v, wanted %v", i, pos[i], want)
		}
	}
}

// The grid holds the aspect ratio, aligns to a whole merged patch, and stays
// inside the token budget. The cases are what the reference's smart resize
// gives, worked through by hand from mtmd-image.cpp.
func TestTargetSizeIsTheSmartResize(t *testing.T) {
	cfg := &VisionConfig{Patch: 16, Merge: 2, MinTokens: 8, MaxTokens: 4096}
	for _, c := range []struct {
		w, h         int
		wantW, wantH int
		why          string
	}{
		{768, 768, 768, 768, "already aligned and inside the budget"},
		{760, 770, 768, 768, "rounded to the nearest 32 either way"},
		{100, 100, 96, 96, "rounds down, still above the floor"},
		// Aligning 16 gives 32x32, whose 1024 pixels fall under the floor of
		// 8192 — so the min branch scales by sqrt(8192/256) and ceils, which
		// lands well above one merged patch rather than on it.
		{16, 16, 96, 96, "under the floor, scaled up until it clears it"},
		{4000, 4000, 2048, 2048, "over the ceiling, scaled back and floored"},
		{6400, 1600, 4096, 1024, "over the ceiling, ratio held at four to one"},
	} {
		gotW, gotH := cfg.TargetSize(c.w, c.h)
		if gotW != c.wantW || gotH != c.wantH {
			t.Errorf("%dx%d gave %dx%d, wanted %dx%d (%s)", c.w, c.h, gotW, gotH, c.wantW, c.wantH, c.why)
		}
		if gotW%32 != 0 || gotH%32 != 0 {
			t.Errorf("%dx%d gave %dx%d, which is not aligned", c.w, c.h, gotW, gotH)
		}
		if n := cfg.Tokens(gotW/cfg.Patch, gotH/cfg.Patch); n < cfg.MinTokens || n > cfg.MaxTokens {
			t.Errorf("%dx%d gives %d tokens, outside [%d, %d]", c.w, c.h, n, cfg.MinTokens, cfg.MaxTokens)
		}
	}
}

// The table is taken untouched at its own side: an interpolation that should
// be the identity is not exactly one, and clip.cpp short-circuits there too.
func TestPositionsAtTheTableSideAreTheTable(t *testing.T) {
	cfg := &VisionConfig{Dim: 2, Merge: 2, PosSide: 2}
	w := &VisionWeights{PosEmbd: []float32{1, 2, 3, 4, 5, 6, 7, 8}}
	got := cfg.Positions(w, 2, 2)
	// Slot order over a 2x2 grid with merge 2 is the raster order itself.
	for i, want := range []float32{1, 2, 3, 4, 5, 6, 7, 8} {
		if got[i] != want {
			t.Fatalf("element %d: %v, wanted %v", i, got[i], want)
		}
	}
}
