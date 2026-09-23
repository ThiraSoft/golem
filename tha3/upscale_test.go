package tha3

import (
	"image"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
	"github.com/ThiraSoft/golem/vk"
)

// openUpscaler reads the upscaler's weights, or skips.
func openUpscaler(t *testing.T) *upscaler {
	t.Helper()
	if !HasEyeUpscaler(Dir()) {
		t.Skipf("no %s in %s: see ref/tha3/README.md", upscalerFile, Dir())
	}
	w, err := openWeights(Dir(), upscalerFile)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.close() })
	u := newUpscaler(w)
	if err := w.err(); err != nil {
		t.Fatal(err)
	}
	return u
}

// The upscaler against the network run in torch by ref/tha3/upscaler.py.
func TestUpscalerMatchesTorch(t *testing.T) {
	u := openUpscaler(t)
	f := loadFixtures(t, "upscaler")
	got := u.forward(f.tensor(t, "in"))
	want := f.tensor(t, "out")
	if got.C != want.C || got.H != want.H || got.W != want.W {
		t.Fatalf("got %dx%dx%d, want %dx%dx%d", got.C, got.H, got.W, want.C, want.H, want.W)
	}
	var worst float64
	for i, v := range got.Data {
		worst = max(worst, float64(abs32(v-want.Data[i])))
	}
	t.Logf("largest difference %.2e", worst)
	if worst > tolerance {
		t.Fatalf("largest difference %.2e, over %.0e", worst, tolerance)
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// The mask covers where the eye alpha is, spreads past it, and fades to
// nothing far from it.
func TestEyeMaskSpreadsAndFades(t *testing.T) {
	alpha := NewTensor(1, 192, 192)
	for y := 90; y < 100; y++ {
		for x := 50; x < 80; x++ {
			alpha.Data[y*192+x] = 1
		}
	}
	m := eyeMask(alpha).Plane(0)
	at := func(y, x int) float32 { return m[y*192+x] }
	if v := at(95, 65); v < 0.99 {
		t.Errorf("middle of the eye: %v, want 1", v)
	}
	if v := at(95, 83); v < 0.3 {
		t.Errorf("just past the eye's corner: %v, want the fade under way", v)
	}
	if v := at(95, 120); v != 0 {
		t.Errorf("far from the eye: %v, want 0", v)
	}
}

// The card's upscaled eyes against the processor's, at a scale of two, on the
// recorded poses: the eyes shut in "wink", the face at rest in "neutral".
func TestCardEyeUpscaleMatchesCPU(t *testing.T) {
	dev, err := vk.Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	dev.Close()
	openUpscaler(t)
	cpu := openPoserOrSkip(t)
	defer cpu.Close()
	card := openPoserOrSkip(t)
	defer card.Close()
	if err := card.UseVulkan(); err != nil {
		t.Fatal(err)
	}
	view := image.Rect(144, 32, 368, 256)
	for _, p := range []*Poser{cpu, card} {
		for _, err := range []error{p.SetView(view), p.SetScale(2), p.SetSharpen(0.8), p.SetEyeUpscale(1)} {
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, run := range []string{"wink", "neutral"} {
		f := loadFixtures(t, run)
		img := f.tensor(t, "image")
		for _, p := range []*Poser{cpu, card} {
			if err := p.SetImage(img); err != nil {
				t.Fatal(err)
			}
		}
		var pose [NumParams]float32
		copy(pose[:], f.Pose)
		want, err := cpu.Pose(pose)
		if err != nil {
			t.Fatal(err)
		}
		got, err := card.Pose(pose)
		if err != nil {
			t.Fatal(err)
		}
		reference.Compare(t, run+" frame", got.Data, want.Data, tolerance)
	}
}
