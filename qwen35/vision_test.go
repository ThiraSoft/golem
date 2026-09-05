package qwen35

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

// A picture with something in it: a gradient with a square, so a tower that
// answered the same thing for every patch would show.
func testPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.RGBA{uint8(x * 255 / w), uint8(y * 255 / h), 64, 255}
			if x > w/3 && x < 2*w/3 && y > h/3 && y < 2*h/3 {
				c = color.RGBA{255, 255, 255, 255}
			}
			im.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, im); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// An image in, one row per output token out, each as wide as the model.
func TestEncodeImageProducesRows(t *testing.T) {
	m, err := Open(qwen38, 256)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.OpenProjector(qwen38mmproj); err != nil {
		t.Skipf("projector: %v", err)
	}

	// 224x224 aligns to 224x224, a grid of 14x14 patches and 49 tokens.
	rows, err := m.EncodeImage(testPNG(t, 224, 224))
	if err != nil {
		t.Fatal(err)
	}
	cfg := m.Vision().Cfg
	w, h := cfg.TargetSize(224, 224)
	want := cfg.Tokens(w/cfg.Patch, h/cfg.Patch)
	if len(rows) != want {
		t.Fatalf("%d rows, wanted %d for a %dx%d grid", len(rows), want, w, h)
	}
	for i, r := range rows {
		if len(r) != m.Cfg.Dim {
			t.Fatalf("row %d is %d wide, wanted the model's %d", i, len(r), m.Cfg.Dim)
		}
	}

	// The rows must differ from one another: a tower that collapsed would
	// return the same vector for every token and still have the right shape.
	same := true
	for i := range rows[0] {
		if rows[0][i] != rows[len(rows)-1][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("every token came back identical, so the tower is not looking at the picture")
	}

	// And they must be finite: a norm with the wrong epsilon or a rotation
	// over the wrong width shows up here first.
	for i, r := range rows {
		for j, v := range r {
			if v != v || v > 1e4 || v < -1e4 {
				t.Fatalf("row %d element %d is %v", i, j, v)
			}
		}
	}
}

// A model without a projector says so rather than answering nothing.
func TestEncodeImageNeedsAProjector(t *testing.T) {
	m, err := Open(qwen38, 256)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if _, err := m.EncodeImage(testPNG(t, 64, 64)); err == nil {
		t.Fatal("an image was encoded without a projector")
	}
}

// The text model's own weights are not a projector.
func TestOpenProjectorRefusesTheTextModel(t *testing.T) {
	m, err := Open(qwen38, 256)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.OpenProjector(qwen38); err == nil {
		t.Fatal("the text model was opened as a projector")
	}
}
