// Package imageio decodes images and puts them in the shape a vision encoder
// reads: planar RGB, one float per channel per pixel.
//
// It knows nothing about any model. What size to resize to is a question the
// model answers; this package answers how.
package imageio

import (
	"fmt"
	"image"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"

	_ "golang.org/x/image/webp"

	"github.com/ThiraSoft/golem/nn"
)

// Image is eight-bit RGB, interleaved, no alpha and no stride: a decoder's
// output flattened to the one layout everything below wants.
type Image struct {
	W, H int
	Pix  []uint8 // W*H*3
}

// Decode reads PNG, JPEG, GIF or WebP. Anything with transparency is composited
// on white, because a model shown a checkerboard of alpha would describe the
// checkerboard.
func Decode(r io.Reader) (*Image, error) {
	src, format, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("imageio: %w", err)
	}
	b := src.Bounds()
	if b.Dx() == 0 || b.Dy() == 0 {
		return nil, fmt.Errorf("imageio: the %s image is empty", format)
	}
	flat := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(flat, flat.Bounds(), image.NewUniform(image.White), image.Point{}, draw.Src)
	draw.Draw(flat, flat.Bounds(), src, b.Min, draw.Over)

	im := &Image{W: b.Dx(), H: b.Dy(), Pix: make([]uint8, b.Dx()*b.Dy()*3)}
	for i, n := 0, im.W*im.H; i < n; i++ {
		im.Pix[3*i+0] = flat.Pix[4*i+0]
		im.Pix[3*i+1] = flat.Pix[4*i+1]
		im.Pix[3*i+2] = flat.Pix[4*i+2]
	}
	return im, nil
}

// PlanarRGB writes the whole red plane, then the green, then the blue, scaled
// to 0..1. That is the layout ggml's convolution reads.
func (im *Image) PlanarRGB(dst []float32) {
	n := im.W * im.H
	if len(dst) != 3*n {
		panic(fmt.Sprintf("imageio: a %dx%d image needs %d floats, given %d", im.W, im.H, 3*n, len(dst)))
	}
	// Three planes of a megapixel is a pass long enough to be worth sharing
	// out, and each pixel is written once by whoever reads it.
	nn.InParallel(n, 3*n*4, func(first, last int) {
		for c := 0; c < 3; c++ {
			plane := dst[c*n : (c+1)*n]
			for i := first; i < last; i++ {
				plane[i] = float32(im.Pix[3*i+c]) / 255
			}
		}
	})
}
