package gemma

import (
	"testing"

	"github.com/ThiraSoft/golem/imageio"
)

func visionCfgForSizing() *VisionConfig {
	return &VisionConfig{
		Dim: 768, Blocks: 16, Heads: 12, HeadDim: 64, FFN: 3072,
		PatchSize: 16, Merge: 3, ProjDim: 1536, Eps: 1e-6,
		RoPEBase: 100, MinTokens: 40, MaxTokens: 280,
	}
}

func TestTargetSizeIsAMultipleOfTheMergedPatch(t *testing.T) {
	cfg := visionCfgForSizing()
	unit := cfg.PatchSize * cfg.Merge // 48
	for _, in := range [][2]int{{1920, 1080}, {31, 17}, {640, 480}, {100, 4000}} {
		w, h := cfg.TargetSize(in[0], in[1])
		if w%unit != 0 || h%unit != 0 {
			t.Errorf("%dx%d became %dx%d, which is not a multiple of %d", in[0], in[1], w, h, unit)
		}
		if w == 0 || h == 0 {
			t.Errorf("%dx%d became %dx%d", in[0], in[1], w, h)
		}
	}
}

func TestTargetSizeStaysWithinTheTokenBudget(t *testing.T) {
	cfg := visionCfgForSizing()
	unit := cfg.PatchSize * cfg.Merge
	for _, in := range [][2]int{{1920, 1080}, {31, 17}, {8000, 8000}, {4000, 100}} {
		w, h := cfg.TargetSize(in[0], in[1])
		tokens := (w / unit) * (h / unit)
		if tokens < cfg.MinTokens || tokens > cfg.MaxTokens {
			t.Errorf("%dx%d became %dx%d, which is %d tokens, outside %d..%d",
				in[0], in[1], w, h, tokens, cfg.MinTokens, cfg.MaxTokens)
		}
	}
}

func TestTargetSizeKeepsTheAspectRatioRoughly(t *testing.T) {
	cfg := visionCfgForSizing()
	w, h := cfg.TargetSize(1600, 400) // 4:1
	got := float64(w) / float64(h)
	if got < 2.5 || got > 6 {
		t.Errorf("a 4:1 image became %dx%d, a ratio of %.2f", w, h, got)
	}
}

func TestFitKeepsTheShapeAndFillsTheRest(t *testing.T) {
	cfg := visionCfgForSizing()
	// A 4:1 image cannot land on a 4.125:1 canvas without a border.
	im := &imageio.Image{W: 1600, H: 400, Pix: make([]uint8, 1600*400*3)}
	for i := range im.Pix {
		im.Pix[i] = 200
	}
	fitted := cfg.Fit(im)
	tw, th := cfg.TargetSize(im.W, im.H)
	if fitted.W != tw || fitted.H != th {
		t.Fatalf("fitted to %dx%d, the canvas is %dx%d", fitted.W, fitted.H, tw, th)
	}
	if fitted.Pix[0] != 0 {
		t.Errorf("the top left corner is %d, expected the pad colour", fitted.Pix[0])
	}
	mid := 3 * ((th/2)*tw + tw/2)
	if fitted.Pix[mid] != 200 {
		t.Errorf("the middle is %d, expected the image", fitted.Pix[mid])
	}
}

func TestPrepareCountsThePatchGrid(t *testing.T) {
	cfg := visionCfgForSizing()
	im := &imageio.Image{W: 640, H: 426, Pix: make([]uint8, 640*426*3)}
	pixels, cols, rows := cfg.Prepare(im)
	if len(pixels) != 3*cols*rows*cfg.PatchSize*cfg.PatchSize {
		t.Fatalf("%d floats for a %dx%d grid of %d-pixel patches", len(pixels), cols, rows, cfg.PatchSize)
	}
	if cols%cfg.Merge != 0 || rows%cfg.Merge != 0 {
		t.Fatalf("a %dx%d grid does not divide by %d", cols, rows, cfg.Merge)
	}
}
