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

func loadVisionFixture(t *testing.T) *visionFixture {
	t.Helper()
	f := loadFixture(t, "vision")
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
	f := loadVisionFixture(t)
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
	f := loadVisionFixture(t)
	tower := openTower(t)
	want := f.tensor(t, "pos_embd")

	pixels, cols, rows := tower.Cfg.Prepare(testImage(t))
	got := tower.Patches(pixels, cols, rows)
	flat := make([]float32, 0, len(got)*tower.Cfg.Dim)
	for _, row := range got {
		flat = append(flat, row...)
	}
	close(t, "patches", flat, want, 5e-3)
}

// Each block is started from the reference's own input, so a divergence is
// attributed to the block it happened in rather than to everything before it.
func TestVisionBlocksMatchTheReference(t *testing.T) {
	f := loadVisionFixture(t)
	tower := openTower(t)
	cfg := tower.Cfg

	_, cols, rows := cfg.Prepare(testImage(t))
	n := cols * rows

	for i := 0; i < cfg.Blocks; i++ {
		name := "pos_embd"
		if i > 0 {
			name = fmt.Sprintf("layer_out-%d", i-1)
		}
		input := f.tensor(t, name)
		xs := make([][]float32, n)
		for p := range xs {
			xs[p] = append([]float32(nil), input[p*cfg.Dim:(p+1)*cfg.Dim]...)
		}
		tower.Block(i, xs, cols)

		flat := make([]float32, 0, n*cfg.Dim)
		for _, row := range xs {
			flat = append(flat, row...)
		}
		closeRelative(t, fmt.Sprintf("block %d", i), flat, f.tensor(t, fmt.Sprintf("layer_out-%d", i)), 1e-3)
	}
}
