package imageio

import "math"

// ResizeBilinear is the resampling clip.cpp performs, and is written to agree
// with it rather than to be the best resampling available: the reference
// samples the source at (i + 0.5) * src/dst - 0.5, clamps that to the edge,
// and mixes the four neighbours. A different rule here would move every
// activation downstream.
func (im *Image) ResizeBilinear(w, h int) *Image {
	out := &Image{W: w, H: h, Pix: make([]uint8, w*h*3)}
	sx := float64(im.W) / float64(w)
	sy := float64(im.H) / float64(h)
	for y := 0; y < h; y++ {
		fy := (float64(y)+0.5)*sy - 0.5
		y0, dy := split(fy, im.H)
		y1 := clampInt(y0+1, 0, im.H-1)
		for x := 0; x < w; x++ {
			fx := (float64(x)+0.5)*sx - 0.5
			x0, dx := split(fx, im.W)
			x1 := clampInt(x0+1, 0, im.W-1)
			for c := 0; c < 3; c++ {
				p00 := float64(im.Pix[3*(y0*im.W+x0)+c])
				p01 := float64(im.Pix[3*(y0*im.W+x1)+c])
				p10 := float64(im.Pix[3*(y1*im.W+x0)+c])
				p11 := float64(im.Pix[3*(y1*im.W+x1)+c])
				top := p00 + (p01-p00)*dx
				bot := p10 + (p11-p10)*dx
				out.Pix[3*(y*w+x)+c] = uint8(math.Round(top + (bot-top)*dy))
			}
		}
	}
	return out
}

// split cuts a source coordinate into the pixel to its left and the weight of
// the one to its right, both clamped inside the image.
func split(f float64, n int) (int, float64) {
	if f < 0 {
		return 0, 0
	}
	i := int(math.Floor(f))
	if i >= n-1 {
		return n - 1, 0
	}
	return i, f - float64(i)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
