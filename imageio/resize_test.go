package imageio

import "testing"

// The grid is corner-aligned: the two ends keep their values exactly and the
// middle is what the reference's truncating interpolation puts between them.
// A two-pixel row stretched to four reads 0, 1/3, 2/3 and 1 of the way across.
func TestResizeBilinearIsCornerAligned(t *testing.T) {
	src := &Image{W: 2, H: 1, Pix: []uint8{0, 0, 0, 100, 100, 100}}
	got := src.ResizeBilinear(4, 1)
	want := []uint8{0, 0, 0, 33, 33, 33, 66, 66, 66, 100, 100, 100}
	for i := range want {
		if got.Pix[i] != want[i] {
			t.Fatalf("pixel %d is %d, expected %d (%v)", i, got.Pix[i], want[i], got.Pix)
		}
	}
}

func TestResizeBilinearKeepsAConstantImage(t *testing.T) {
	src := &Image{W: 3, H: 3, Pix: make([]uint8, 27)}
	for i := range src.Pix {
		src.Pix[i] = 42
	}
	got := src.ResizeBilinear(7, 5)
	for i, v := range got.Pix {
		if v != 42 {
			t.Fatalf("pixel %d of a constant image came out %d", i, v)
		}
	}
}

func TestResizeToTheSameSizeCopies(t *testing.T) {
	src := &Image{W: 2, H: 1, Pix: []uint8{1, 2, 3, 4, 5, 6}}
	got := src.ResizeBilinear(2, 1)
	for i := range src.Pix {
		if got.Pix[i] != src.Pix[i] {
			t.Fatalf("Pix = %v, expected %v", got.Pix, src.Pix)
		}
	}
}

func TestPadIntoCentresAndFills(t *testing.T) {
	src := &Image{W: 2, H: 1, Pix: []uint8{10, 10, 10, 20, 20, 20}}
	got := src.PadInto(4, 3, [3]uint8{0, 0, 0})
	// Offsets round down: x by one, y by one.
	if got.W != 4 || got.H != 3 {
		t.Fatalf("padded to %dx%d", got.W, got.H)
	}
	at := func(x, y, c int) uint8 { return got.Pix[3*(y*got.W+x)+c] }
	if at(1, 1, 0) != 10 || at(2, 1, 0) != 20 {
		t.Fatalf("the image did not land at (1,1): %v", got.Pix)
	}
	if at(0, 0, 0) != 0 || at(3, 2, 0) != 0 {
		t.Fatal("the border was not filled")
	}
}

func TestPlanarRGBIsChannelMajor(t *testing.T) {
	im := &Image{W: 2, H: 1, Pix: []uint8{255, 0, 0, 0, 255, 0}}
	dst := make([]float32, 6)
	im.PlanarRGB(dst)
	want := []float32{1, 0, 0, 1, 0, 0}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst = %v, expected %v", dst, want)
		}
	}
}
