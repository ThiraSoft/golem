package gemma

// Reading what llama.cpp recorded for one image. The recording is not
// committed — ref/gemma/vision.run writes it in one command — and a machine
// without it skips.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/imageio"
)

// visionFixture is the recording plus what the text side has to agree with:
// where the picture sits in the prompt and how many tokens it became.
type visionFixture struct {
	*fixture
	ImageStart   int `json:"image_start"`
	NImageTokens int `json:"n_image_tokens"`
}

// loadVisionFixture reads one of the vision recordings: "vision" for E2B's
// tower, "vision12" for the 12B's embedder.
func loadVisionFixture(t *testing.T, name string) *visionFixture {
	t.Helper()
	f := loadFixture(t, name)
	v := &visionFixture{fixture: f}
	raw, err := os.ReadFile(filepath.Join(f.dir, "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
	return v
}

// testImage is the picture the recording was made from.
func testImage(t *testing.T) *imageio.Image {
	t.Helper()
	f, err := os.Open(filepath.Join(repoRoot(t), "testdata", "gemma", "shapes.png"))
	if err != nil {
		t.Skipf("the test image is not on this machine: %v", err)
	}
	defer f.Close()
	im, err := imageio.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return im
}

// openTower builds the tower from the projector named by GOLEM_MMPROJ.
func openTower(t *testing.T) *VisionTower {
	t.Helper()
	g := openMMProj(t)
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

// close compares against the reference within an absolute tolerance, and says
// where the first disagreement is rather than that there was one.
func close(t *testing.T, what string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: this engine produced %d values, the reference %d", what, len(got), len(want))
	}
	for i := range got {
		if diff := got[i] - want[i]; diff > tol || diff < -tol {
			t.Fatalf("%s: element %d is %g, the reference has %g", what, i, got[i], want[i])
		}
	}
}

// closeRelative is the same comparison against a tolerance scaled to what the
// tensor holds, which is what the tower's later blocks need: the residual
// stream grows from a range of sixty in the first block to eighteen hundred in
// the last, and an absolute bound tight enough for the first would be a
// thousand times too tight for the last. The rounding itself does not grow —
// every block below stays under a thousandth of its own range.
func closeRelative(t *testing.T, what string, got, want []float32, fraction float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: this engine produced %d values, the reference %d", what, len(got), len(want))
	}
	var scale float32
	for _, v := range want {
		if a := abs32(v); a > scale {
			scale = a
		}
	}
	tol := fraction * scale
	for i := range got {
		if diff := abs32(got[i] - want[i]); diff > tol {
			t.Fatalf("%s: element %d is %g, the reference has %g — %g apart, and %g of a range of %g is allowed",
				what, i, got[i], want[i], diff, tol, scale)
		}
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// The pixels first. A resize that differs makes every later comparison
// meaningless, so this is the test that runs before the others are believed.
func TestVisionPixelsMatchTheReference(t *testing.T) {
	f := loadVisionFixture(t, "vision")
	tower := openTower(t)
	want := f.tensor(t, "inp_raw_scaled")

	pixels, cols, rows := tower.Cfg.Prepare(testImage(t))
	if len(want) != len(pixels) {
		t.Fatalf("the reference resized to %d values, this engine to %d (%dx%d patches)",
			len(want), len(pixels), cols, rows)
	}
	scaled := make([]float32, len(pixels))
	for i, v := range pixels {
		scaled[i] = v*2 - 1 // the graph's scale_bias(2, -1)
	}
	close(t, "pixels", scaled, want, 1e-5)
}

func TestVisionPatchesMatchTheReference(t *testing.T) {
	f := loadVisionFixture(t, "vision")
	tower := openTower(t)
	want := f.tensor(t, "pos_embd")

	pixels, cols, rows := tower.Cfg.Prepare(testImage(t))
	close(t, "patches", tower.Patches(pixels, cols, rows), want, 5e-3)
}

// Each block is started from the reference's own input, so a divergence is
// attributed to the block it happened in rather than to everything before it.
func TestVisionBlocksMatchTheReference(t *testing.T) {
	f := loadVisionFixture(t, "vision")
	tower := openTower(t)
	cfg := tower.Cfg

	_, cols, rows := cfg.Prepare(testImage(t))
	s := tower.takeScratch(cols * rows)

	for i := 0; i < cfg.Blocks; i++ {
		name := "pos_embd"
		if i > 0 {
			name = fmt.Sprintf("layer_out-%d", i-1)
		}
		xs := append([]float32(nil), f.tensor(t, name)...)
		tower.Block(i, xs, cols, s)
		closeRelative(t, fmt.Sprintf("block %d", i), xs, f.tensor(t, fmt.Sprintf("layer_out-%d", i)), 5e-4)
	}
}

// The pooler and the projector, each started from the reference's own input,
// which is what separates their arithmetic from what the blocks before them
// accumulated.
func TestVisionPoolAndProjectMatchTheReference(t *testing.T) {
	f := loadVisionFixture(t, "vision")
	tower := openTower(t)
	cfg := tower.Cfg
	_, cols, rows := cfg.Prepare(testImage(t))

	last := append([]float32(nil), f.tensor(t, fmt.Sprintf("layer_out-%d", cfg.Blocks-1))...)
	closeRelative(t, "pooled", tower.Pool(last, cols, rows), f.tensor(t, "pooled"), 1e-4)

	projected := tower.Project(append([]float32(nil), f.tensor(t, "pooled")...))
	flat := make([]float32, 0, len(projected)*cfg.ProjDim)
	for _, r := range projected {
		flat = append(flat, r...)
	}
	closeRelative(t, "projected", flat, f.tensor(t, "projected"), 1e-3)
}

// And the whole tower at once. The bar is twenty times looser than any single
// stage's, and that factor is measured rather than chosen: a block started
// from the reference's own input lands within a ten-thousandth of its range,
// and letting sixteen of them feed one another compounds that to eight
// thousandths by the last. Nothing in the tower rounds more than ggml does —
// the products take their activation in bfloat16, as ggml's kernel for a
// bfloat16 weight does — so what is left is the order the sums are taken in,
// and a residual stack sixteen deep is what turns that into a visible number.
//
// The proof that eight thousandths is small enough to be nothing is not here:
// it is TestVisionGenerationMatchesTheReference, which draws the same tokens.
func TestVisionEncodeMatchesTheReference(t *testing.T) {
	f := loadVisionFixture(t, "vision")
	tower := openTower(t)
	want := f.tensor(t, "projected")

	got := tower.Encode(testImage(t))
	if len(got) != f.NImageTokens {
		t.Fatalf("this engine made %d tokens, llama.cpp %d", len(got), f.NImageTokens)
	}
	if len(got) < tower.Cfg.MinTokens || len(got) > tower.Cfg.MaxTokens {
		t.Fatalf("%d tokens is outside %d..%d", len(got), tower.Cfg.MinTokens, tower.Cfg.MaxTokens)
	}
	flat := make([]float32, 0, len(got)*tower.Cfg.ProjDim)
	for _, row := range got {
		flat = append(flat, row...)
	}
	closeRelative(t, "the whole tower", flat, want, 1e-2)
}

// The whole path: the picture through the tower, the rows into the prompt, the
// span attended to in both directions, and the same tokens out as llama.cpp
// drew from the same prompt.
//
// This is what says the eight thousandths the tower ends up from the reference
// are nothing that matters: an argmax either agrees or it does not. The tie
// rule is TestGreedyGenerationMatchesTheReference's, for the same reason and
// with the same number — the logits of this engine and of llama.cpp are known
// to sit a fraction of a logit apart, and a choice decided inside that is not
// a fault.
func TestVisionGenerationMatchesTheReference(t *testing.T) {
	f := loadVisionFixture(t, "vision")
	if os.Getenv("GOLEM_MMPROJ") == "" {
		t.Skip("set GOLEM_MMPROJ to run this test")
	}
	m := openTextModel(t)
	if err := m.OpenProjector(os.Getenv("GOLEM_MMPROJ")); err != nil {
		t.Fatal(err)
	}
	rows, err := m.EncodeImage(testImageBytes(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != f.NImageTokens {
		t.Fatalf("this engine made %d image tokens, llama.cpp %d", len(rows), f.NImageTokens)
	}

	// The prompt is the fixture's own text tokens, which keeps this test about
	// the picture rather than about the template. The recorder wrote the
	// padding token where the soft tokens went, so what is left after taking
	// them out is the text, markers included, and BuildPrompt fills the
	// markers back in.
	var text []int32
	text = append(text, f.Tokens[:f.ImageStart]...)
	text = append(text, f.Tokens[f.ImageStart+f.NImageTokens:]...)
	p, err := m.BuildPrompt(text, [][][]float32{rows}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Tokens) != len(f.Tokens) {
		t.Fatalf("the prompt came to %d tokens, llama.cpp's to %d", len(p.Tokens), len(f.Tokens))
	}
	for i := range p.Tokens {
		if p.Embeds[i] == nil && p.Tokens[i] != f.Tokens[i] {
			t.Fatalf("token %d is %d, llama.cpp's is %d", i, p.Tokens[i], f.Tokens[i])
		}
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
