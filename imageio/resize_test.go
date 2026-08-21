package imageio

import "testing"

// A two-by-two image doubled: the corners keep their values and the middle is
// what bilinear interpolation puts between them. The coordinate rule is the
// one clip.cpp uses — the source position of output pixel i is
// (i + 0.5) * srcW/dstW - 0.5, clamped to the edge.
func TestResizeBilinearDoubling(t *testing.T) {
	src := &Image{W: 2, H: 1, Pix: []uint8{0, 0, 0, 100, 100, 100}}
	got := src.ResizeBilinear(4, 1)
	want := []uint8{0, 0, 0, 25, 25, 25, 75, 75, 75, 100, 100, 100}
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
