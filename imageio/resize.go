package imageio

// ResizeBilinear is the resampling clip.cpp performs, and is written to agree
// with it rather than to be the best resampling available.
//
// Two details are the whole point. The grid is corner-aligned — output pixel i
// reads source position i*(src-1)/(dst-1), so the first and last output pixels
// are exactly the first and last input ones — and the result is truncated
// rather than rounded, because the reference casts a float to a byte. Neither
// is what a graphics library would do, and both are what the model was fed.
func (im *Image) ResizeBilinear(w, h int) *Image {
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	out := &Image{W: w, H: h, Pix: make([]uint8, w*h*3)}
	if im.W == w && im.H == h {
		copy(out.Pix, im.Pix)
		return out
	}
	var xRatio, yRatio float32
	if w > 1 {
		xRatio = float32(im.W-1) / float32(w-1)
	}
	if h > 1 {
		yRatio = float32(im.H-1) / float32(h-1)
	}
	for y := 0; y < h; y++ {
		py := float32(y) * yRatio
		y0 := minInt(int(py), im.H-1)
		y1 := minInt(y0+1, im.H-1)
		yf := py - float32(y0)
		for x := 0; x < w; x++ {
			px := float32(x) * xRatio
			x0 := minInt(int(px), im.W-1)
			x1 := minInt(x0+1, im.W-1)
			xf := px - float32(x0)
			for c := 0; c < 3; c++ {
				p00 := float32(im.Pix[3*(y0*im.W+x0)+c])
				p10 := float32(im.Pix[3*(y0*im.W+x1)+c])
				p01 := float32(im.Pix[3*(y1*im.W+x0)+c])
				p11 := float32(im.Pix[3*(y1*im.W+x1)+c])
				top := lerp(p00, p10, xf)
				bottom := lerp(p01, p11, xf)
				out.Pix[3*(y*w+x)+c] = uint8(lerp(top, bottom, yf))
			}
		}
	}
	return out
}

// PadInto centres this image on a canvas of w by h filled with one colour, and
// is what a resize that preserves the aspect ratio exactly needs: the shape is
// kept and the difference becomes a border. The offsets round down, as the
// reference's integer division does.
func (im *Image) PadInto(w, h int, colour [3]uint8) *Image {
	out := &Image{W: w, H: h, Pix: make([]uint8, w*h*3)}
	for i := 0; i < w*h; i++ {
		out.Pix[3*i+0] = colour[0]
		out.Pix[3*i+1] = colour[1]
		out.Pix[3*i+2] = colour[2]
	}
	ox, oy := (w-im.W)/2, (h-im.H)/2
	for y := 0; y < im.H; y++ {
		dy := y + oy
		if dy < 0 || dy >= h {
			continue
		}
		for x := 0; x < im.W; x++ {
			dx := x + ox
			if dx < 0 || dx >= w {
				continue
			}
			copy(out.Pix[3*(dy*w+dx):3*(dy*w+dx)+3], im.Pix[3*(y*im.W+x):3*(y*im.W+x)+3])
		}
	}
	return out
}

func lerp(a, b, t float32) float32 { return a + (b-a)*t }

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
