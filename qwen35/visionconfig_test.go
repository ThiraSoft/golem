package qwen35

import (
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

const qwen38mmproj = "/mnt/data/LLMs_models/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf"

func TestVisionConfigReadsTheProjector(t *testing.T) {
	g, err := tensors.OpenGGUF(qwen38mmproj)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer g.Close()
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"blocks", cfg.Blocks, 27},
		{"dim", cfg.Dim, 1152},
		{"heads", cfg.Heads, 16},
		{"head dim", cfg.HeadDim, 72},
		{"ffn", cfg.FFN, 4304},
		{"patch", cfg.Patch, 16},
		{"merge", cfg.Merge, 2},
		{"positions a side", cfg.PosSide, 48},
		{"projection", cfg.ProjDim, 5120},
	} {
		if c.got != c.want {
			t.Errorf("%s: %d, wanted %d", c.name, c.got, c.want)
		}
	}
}

// The merger folds a square, so an image is worth a quarter of its patches.
func TestTokensFoldTheSquare(t *testing.T) {
	cfg := &VisionConfig{Merge: 2}
	if got := cfg.Tokens(48, 48); got != 576 {
		t.Errorf("48x48 gives %d tokens, wanted 576", got)
	}
}

// A file that is not this projector is refused by name rather than read into
// the wrong shapes.
func TestVisionConfigRefusesAnotherFile(t *testing.T) {
	g, err := tensors.OpenGGUF(qwen38)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer g.Close()
	if _, err := LoadVisionConfig(g); err == nil {
		t.Fatal("the text model was read as a projector")
	}
}

// Every tensor the tower computes with binds at the shape the config expects.
// A matrix bound at the wrong width does not fail — it reads a neighbouring
// row and answers something plausible — so the shapes are checked at load and
// this is what says the checking runs.
func TestVisionWeightsBind(t *testing.T) {
	g, err := tensors.OpenGGUF(qwen38mmproj)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer g.Close()
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Blocks) != cfg.Blocks {
		t.Fatalf("%d blocks, wanted %d", len(w.Blocks), cfg.Blocks)
	}
	if got := w.Blocks[0].QKV.W.Rows; got != 3*cfg.Dim {
		t.Errorf("fused qkv gives %d rows, wanted %d", got, 3*cfg.Dim)
	}
	if got := w.MM2.W.Rows; got != cfg.ProjDim {
		t.Errorf("the merger lands on %d, wanted the model's %d", got, cfg.ProjDim)
	}
	if got := len(w.PosEmbd); got != cfg.Dim*cfg.PosSide*cfg.PosSide {
		t.Errorf("the position table holds %d floats", got)
	}
	// The two patch convolutions are the same shape and are not the same
	// tensor: a Conv3d of depth two split in half.
	if w.PatchA.Rows != w.PatchB.Rows || w.PatchA.Cols != w.PatchB.Cols {
		t.Error("the two patch convolutions differ in shape")
	}
	if &w.PatchA.Data[0] == &w.PatchB.Data[0] {
		t.Error("the two patch convolutions are the same bytes")
	}
}
