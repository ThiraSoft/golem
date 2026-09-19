package tha3

import (
	"image"
	"image/color"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
)

func TestImageRoundTrip(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	img.SetNRGBA(0, 0, color.NRGBA{200, 100, 10, 255})
	img.SetNRGBA(1, 0, color.NRGBA{255, 255, 255, 0})
	x := FromNRGBA(img)
	for c := 0; c < 4; c++ {
		if v := x.Plane(c)[1]; v != -1 {
			t.Fatalf("transparent pixel channel %d = %v, want -1", c, v)
		}
	}
	back := ToNRGBA(x)
	if got := back.NRGBAAt(0, 0); got != (color.NRGBA{200, 100, 10, 255}) {
		t.Fatalf("round trip gave %v", got)
	}
	if got := back.NRGBAAt(1, 0); got.A != 0 {
		t.Fatalf("transparent pixel came back with alpha %d", got.A)
	}
}

// The picture the recordings were made from must load to the same tensor
// the demo's own loader produced.
func TestLoadImageMatchesDemo(t *testing.T) {
	f := loadFixtures(t, "neutral")
	path := os.Getenv("THA3_PICTURE")
	if path == "" {
		t.Skip("THA3_PICTURE is not set: the picture dump.py was run on")
	}
	got, err := LoadImage(path)
	if err != nil {
		t.Fatal(err)
	}
	want := f.tensor(t, "image")
	reference.Compare(t, "image", got.Data, want.Data, 1e-6)
}
