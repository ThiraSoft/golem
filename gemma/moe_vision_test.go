package gemma

// The 26B A4B's pictures.
//
// The tower does not depend on the language model behind it, and the point of
// this file is to say so with numbers rather than to hope: the same graph, the
// same node names, a different checkpoint. Nothing in gemma/vision*.go was
// written for this model, and nothing had to be.

import (
	"fmt"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

func mmproj26BPath(tb testing.TB) string {
	tb.Helper()
	path := os.Getenv("GOLEM_MMPROJ_26B")
	if path == "" {
		tb.Skip("set GOLEM_MMPROJ_26B to the 26B's projector to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("GOLEM_MMPROJ_26B names %s, which is not there", path)
	}
	return path
}

// TestMoEVisionTowerMatchesTheReference compares the tower's last node — the
// rows the language model is handed — against llama.cpp's.
func TestMoEVisionTowerMatchesTheReference(t *testing.T) {
	f := loadVisionFixture(t, "vision26")
	path := mmproj26BPath(t)

	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Unified {
		t.Fatalf("the 26B's projector declares gemma4uv; this test was written for a tower")
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	tower := NewVisionTower(cfg, w)

	rows := tower.Encode(testImage(t))
	want := f.tensor(t, "projected")
	flat := make([]float32, 0, len(rows)*len(rows[0]))
	for _, r := range rows {
		flat = append(flat, r...)
	}
	if len(flat) != len(want) {
		t.Fatalf("this engine made %d values, the reference %d", len(flat), len(want))
	}
	// This tower ends up further from the reference than the E2B's does:
	// thirty-six thousandths of the range against eight. It is depth, and the
	// measurement is in TestMoEVisionBlocks and TestMoEVisionStages below —
	// every block on its own lands within two ten-thousandths, the same as
	// E2B's do, the pooling is exact to a part in ten million, and what is
	// left is twenty-seven residual blocks feeding one another where E2B has
	// sixteen. Run the same instrument over E2B and the curve has the same
	// shape, ending at eight thousandths instead of twenty-eight.
	//
	// What it costs is settled by the only test that can settle it:
	// TestMoEVisionGeneration draws llama.cpp's own answer about the picture,
	// token for token.
	closeRelative(t, "the whole tower", flat, want, 5e-2)
}

// TestMoEVisionGeneration is what the tower is for: the reference's own answer
// about the picture, token for token.
func TestMoEVisionGeneration(t *testing.T) {
	f := loadVisionFixture(t, "vision26")
	path := model26BPathT(t)
	proj := mmproj26BPath(t)

	m, err := Open(path, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.OpenProjector(proj); err != nil {
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
	p, err := m.BuildPrompt(text, [][][]float32{rows}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// The same tie rule as everywhere else here, and looser by the amount the
	// mixture is: thirty blocks of eight quantized expert products sit further
	// from the reference than a dense stack does.
	const tie = 1.5
	hidden := m.ForwardPrompt(p, 0)
	last := hidden[len(hidden)-1]
	logits := make([]float32, m.Cfg.Vocab)
	pos := len(p.Tokens)
	for step, want := range f.Greedy {
		m.Logits(last, logits)
		if got := Argmax(logits); got != want {
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

// TestMoEVisionBlocks starts each block from the reference's own input, so a
// divergence is attributed to the block it happened in rather than to
// everything before it. This is the test that would catch a step this engine
// had missed; that it passes at a bar E2B's blocks also hold to is why the
// whole tower's looser number is depth and not a bug.
func TestMoEVisionBlocks(t *testing.T) {
	f := loadVisionFixture(t, "vision26")
	g, err := tensors.OpenGGUF(mmproj26BPath(t))
	if err != nil {
		t.Fatal(err)
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
	tower := NewVisionTower(cfg, w)

	_, cols, rows := cfg.Prepare(testImage(t))
	s := tower.takeScratch(cols * rows)
	for i := 0; i < cfg.Blocks; i++ {
		name := "pos_embd"
		if i > 0 {
			name = fmt.Sprintf("layer_out-%d", i-1)
		}
		xs := append([]float32(nil), f.tensor(t, name)...)
		tower.Block(i, xs, cols, s)
		closeRelative(t, fmt.Sprintf("block %d", i), xs,
			f.tensor(t, fmt.Sprintf("layer_out-%d", i)), 1e-3)
	}
}

// TestMoEVisionPoolAndProject takes each of the last two stages from the
// reference's own input, which is what separates their arithmetic from what
// the blocks accumulated.
//
// The projection is a few parts in a hundred thousand where E2B's is a part in
// ten million, and the difference is the one step E2B does not have: this
// projector standardises before the norm, and subtracting a bias from a value
// near it loses digits. llama.cpp subtracts in float32 too and loses the same
// ones. Beside the twenty-eight thousandths the blocks bring, it is nothing.
func TestMoEVisionPoolAndProject(t *testing.T) {
	f := loadVisionFixture(t, "vision26")
	g, err := tensors.OpenGGUF(mmproj26BPath(t))
	if err != nil {
		t.Fatal(err)
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
	tower := NewVisionTower(cfg, w)
	_, cols, rows := cfg.Prepare(testImage(t))

	// The tower's last output. "result_norm" in this recording belongs to the
	// text model and is 2816 wide, which is a trap worth one line of comment.
	last := append([]float32(nil), f.tensor(t, fmt.Sprintf("layer_out-%d", cfg.Blocks-1))...)
	closeRelative(t, "pooled", tower.Pool(last, cols, rows), f.tensor(t, "pooled"), 1e-6)

	projected := tower.Project(append([]float32(nil), f.tensor(t, "pooled")...))
	flat := make([]float32, 0, len(projected)*cfg.ProjDim)
	for _, r := range projected {
		flat = append(flat, r...)
	}
	closeRelative(t, "projected", flat, f.tensor(t, "projected"), 1e-3)
}

// TestMoEVisionStages walks the tower one stage at a time, so that a gap at
// the end says which stage opened it. The resize and the patch embedding are
// where a divergence would be cheapest to find and worst to miss: every later
// comparison is meaningless if the picture went in differently.
func TestMoEVisionStages(t *testing.T) {
	f := loadVisionFixture(t, "vision26")
	g, err := tensors.OpenGGUF(mmproj26BPath(t))
	if err != nil {
		t.Fatal(err)
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
	tower := NewVisionTower(cfg, w)

	pixels, cols, rows := cfg.Prepare(testImage(t))
	scaled := make([]float32, len(pixels))
	for i, v := range pixels {
		scaled[i] = v*2 - 1
	}
	closeRelative(t, "pixels", scaled, f.tensor(t, "inp_raw_scaled"), 1e-3)
	closeRelative(t, "patches", tower.Patches(pixels, cols, rows), f.tensor(t, "pos_embd"), 1e-2)

}
