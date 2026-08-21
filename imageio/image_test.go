package imageio

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

func TestDecodePNG(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 2, 1))
	src.Set(0, 0, color.RGBA{R: 255, A: 255})
	src.Set(1, 0, color.RGBA{G: 255, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	im, err := Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if im.W != 2 || im.H != 1 {
		t.Fatalf("decoded a %dx%d image", im.W, im.H)
	}
	want := []uint8{255, 0, 0, 0, 255, 0}
	for i := range want {
		if im.Pix[i] != want[i] {
			t.Fatalf("Pix = %v, expected %v", im.Pix, want)
		}
	}
}

func TestDecodeRefusesRubbish(t *testing.T) {
	if _, err := Decode(bytes.NewReader([]byte("not an image"))); err == nil {
		t.Fatal("decoding rubbish returned no error")
	}
}
