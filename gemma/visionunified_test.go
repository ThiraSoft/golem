package gemma

// The 12B's projector, against llama.cpp's own recording of it.
//
// It is a second architecture rather than a second size: the file declares
// gemma4uv, carries eleven tensors and no block at all, and what it computes
// is an embedder whose output goes straight into the language model. The
// recording is testdata/gemma/vision12, written by ref/gemma/vision12.run, and
// it holds the two nodes that graph names — the patches with their positions
// added, and the projection.

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

// openMMProj12 is the 12B's projector, named by GOLEM_MMPROJ_12B. It is not
// the same file as GOLEM_MMPROJ and the two are not interchangeable.
func openMMProj12(tb testing.TB) *tensors.GGUF {
	tb.Helper()
	path := os.Getenv("GOLEM_MMPROJ_12B")
	if path == "" {
		tb.Skip("set GOLEM_MMPROJ_12B to a Gemma 4 12B mmproj GGUF to run this test")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		tb.Fatalf("GOLEM_MMPROJ_12B names %s, which will not open: %v", path, err)
	}
	tb.Cleanup(func() { g.Close() })
	return g
}

func openUnifiedTower(t *testing.T) *VisionTower {
	t.Helper()
	g := openMMProj12(t)
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return NewVisionTower(cfg, w)
}

func TestLoadUnifiedVisionConfig(t *testing.T) {
	cfg, err := LoadVisionConfig(openMMProj12(t))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Unified {
		t.Fatal("the 12B's projector was not read as a unified one")
	}
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"dim", cfg.Dim, 3840},
		{"blocks", cfg.Blocks, 0},
		// The merge is folded into the patch: 16 times 3, and then nothing to
		// pool afterwards.
		{"patch", cfg.PatchSize, 48},
		{"merge", cfg.Merge, 1},
		{"projection", cfg.ProjDim, 3840},
		{"min tokens", cfg.MinTokens, 40},
		{"max tokens", cfg.MaxTokens, 280},
	} {
		if c.got != c.want {
			t.Errorf("%s is %d, expected %d", c.name, c.got, c.want)
		}
	}
}

// The patches, embedded, normed and given their positions — the node
// llama.cpp names pos_embd, which is the last one before the norm that closes
// the embedder.
func TestUnifiedPatchesMatchTheReference(t *testing.T) {
	f := loadFixture(t, "vision12")
	tower := openUnifiedTower(t)
	want := f.tensor(t, "pos_embd")

	pixels, cols, rows := tower.Cfg.Prepare(testImage(t))
	got := tower.embedPatches(pixels, cols, rows)
	if len(got) != len(want) {
		t.Fatalf("this engine made %d patch values, the reference %d (%dx%d patches)",
			len(got), len(want), cols, rows)
	}
	closeRelative(t, "pos_embd", got, want, 1e-3)
}

func TestUnifiedProjectionMatchesTheReference(t *testing.T) {
	f := loadFixture(t, "vision12")
	tower := openUnifiedTower(t)
	want := f.tensor(t, "projected")

	rows := tower.Encode(testImage(t))
	if len(rows)*tower.Cfg.ProjDim != len(want) {
		t.Fatalf("this engine made %d soft tokens, the reference %d",
			len(rows), len(want)/tower.Cfg.ProjDim)
	}
	got := make([]float32, 0, len(want))
	for _, row := range rows {
		got = append(got, row...)
	}
	closeRelative(t, "projected", got, want, 1e-3)
}

// And the whole path, the way TestVisionGenerationMatchesTheReference does it
// for E2B: the picture encoded, spliced into the fixture's own tokens, and the
// same continuation drawn from it.
func TestUnifiedVisionGenerationMatchesTheReference(t *testing.T) {
	f := loadVisionFixture(t, "vision12")
	if os.Getenv("GOLEM_MMPROJ_12B") == "" {
		t.Skip("set GOLEM_MMPROJ_12B to run this test")
	}
	m, err := Open(model12BPath(t), 4096)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	if err := m.OpenVision(os.Getenv("GOLEM_MMPROJ_12B")); err != nil {
		t.Fatal(err)
	}
	rows, err := m.EncodeImage(testImageBytes(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != f.NImageTokens {
		t.Fatalf("this engine made %d image tokens, llama.cpp %d", len(rows), f.NImageTokens)
	}

	var text []int32
	text = append(text, f.Tokens[:f.ImageStart]...)
	text = append(text, f.Tokens[f.ImageStart+f.NImageTokens:]...)
	p, err := m.BuildPrompt(text, [][][]float32{rows})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Tokens) != len(f.Tokens) {
		t.Fatalf("the prompt came to %d tokens, llama.cpp's to %d", len(p.Tokens), len(f.Tokens))
	}
	const tie = 0.5
	hidden := m.ForwardPrompt(p, 0)
	last := hidden[len(hidden)-1]
	logits := make([]float32, m.Cfg.Vocab)
	pos := len(p.Tokens)
	for step, want := range f.Greedy {
		m.Logits(last, logits)
		got := Argmax(logits)
		if got != want {
			if margin := logits[got] - logits[want]; margin > tie {
				t.Fatalf("step %d: chose %d over the reference's %d by %v, which is past a tie",
					step, got, want, margin)
			} else {
				t.Logf("step %d: chose %d over %d by %v, a tie inside the measured gap",
					step, got, want, margin)
			}
		}
		last = m.Forward(want, pos)
		pos++
	}
}
