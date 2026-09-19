package tha3

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/ThiraSoft/golem/nn"
)

// The networks see 512×512 and nothing larger, but most of what they do is
// decide: where each pixel of the picture goes, and where to paint over it.
// At a scale above one, every such decision is resized and applied again to
// a picture that many times larger, so that what the networks only move
// keeps that picture's detail. What they paint, the inside of the mouth or
// a closed eye, is as blurred as it was at 512.

// maxScale bounds the larger picture: at 8 it is 4096 on a side, 256 MB.
const maxScale = 8

// SetScale makes Pose return its view k times larger, drawn from the larger
// picture SetImageHigh gives. A scale of one is the frame the networks make.
// A picture already set is kept, resized; the larger one it had is not, since
// it is the wrong size. On the card a new scale rebuilds the passes, so it
// is set once, early, like the view.
func (p *Poser) SetScale(k int) error {
	if p.closed {
		return errClosed
	}
	if k < 1 || k > maxScale {
		return fmt.Errorf("tha3: scale %d is not between 1 and %d", k, maxScale)
	}
	if k == p.scale {
		return nil
	}
	old := p.scale
	p.scale = k
	p.haveLast = false
	if p.gpu != nil {
		p.gpu.close()
		p.gpu = nil
		// The card is built without a picture: the one set, if any, is
		// sent below with its larger picture at the new size.
		image := p.image
		p.image = Tensor{}
		err := p.useVulkan(p.gpuTrace)
		p.image = image
		if err != nil {
			p.scale = old
			return fmt.Errorf("tha3: the card for scale %d: %w", k, err)
		}
	}
	if p.image.Data == nil {
		p.high = Tensor{}
		return nil
	}
	return p.SetImageHigh(p.image, Tensor{})
}

// resize is ResizeBilinear, or x itself when it is the size already.
func resize(x Tensor, h, w int) Tensor {
	if x.H == h && x.W == w {
		return x
	}
	return ResizeBilinear(x, h, w)
}

// Zoom frames part of the view: Scale times larger, centred at (X, Y) as
// fractions of the view's width and height. The zero Zoom is the whole view.
type Zoom struct{ Scale, X, Y float64 }

func (z Zoom) orWhole() Zoom {
	if z.Scale == 0 {
		return Zoom{1, 0.5, 0.5}
	}
	return z
}

// zoomAt is where the zoom goes in the card's pose buffer, after the pose.
const zoomAt = NumParams

// SetBackground makes the poser finish its frames for the screen: laid on
// bg, in sRGB, framed by a zoom, as PoseRGBA returns them. On the card that
// is the last step of the editor, so that only bytes come back, and none of
// it is left for the processor. Pose no longer works afterwards. On the card
// it rebuilds the passes, so it is set once, early, like the view.
func (p *Poser) SetBackground(bg color.RGBA) error {
	if p.closed {
		return errClosed
	}
	c := [3]uint8{bg.R, bg.G, bg.B}
	if p.finish && c == p.bg {
		return nil
	}
	oldFinish, oldBg := p.finish, p.bg
	p.finish, p.bg = true, c
	p.haveLast = false
	if p.gpu == nil {
		return nil
	}
	p.gpu.close()
	p.gpu = nil
	if err := p.useVulkan(p.gpuTrace); err != nil {
		p.finish, p.bg = oldFinish, oldBg
		return fmt.Errorf("tha3: the card for the background: %w", err)
	}
	return nil
}

// PoseRGBA is Pose finished: the view, as large as the scale makes it, framed
// by zoom and laid on the background SetBackground gave. The pose is cached
// as Pose's is; a new zoom alone runs only the last stage.
func (p *Poser) PoseRGBA(pose [NumParams]float32, zoom Zoom) (*image.RGBA, error) {
	if !p.finish {
		return nil, fmt.Errorf("tha3: PoseRGBA before SetBackground")
	}
	frame, err := p.render(pose, zoom.orWhole())
	if err != nil {
		return nil, err
	}
	img := image.NewRGBA(image.Rect(0, 0, frame.W, frame.H))
	for i, v := range frame.Data {
		u := uint32(v)
		img.Pix[4*i], img.Pix[4*i+1], img.Pix[4*i+2], img.Pix[4*i+3] = uint8(u), uint8(u>>8), uint8(u>>16), 255
	}
	return img, nil
}

// finishCPU is the card's EditHighRGB on the processor, the same arithmetic
// in the same order: high is the picture at the scale, face the morphed face
// on it.
func (p *Poser) finishCPU(high, face Tensor) Tensor {
	k := high.H / Size
	v := p.view
	winY, winX, winH, winW := float32(v.Min.Y*k), float32(v.Min.X*k), v.Dy()*k, v.Dx()*k
	outH, outW := winH, winW
	z := p.zoom.orWhole()
	s, cx, cy := float32(z.Scale), float32(z.X), float32(z.Y)
	e := p.edited
	lowH, lowW := e.grid.H, e.grid.W
	h, w := high.H, high.W
	fy0, fx0 := 32*k, 160*k
	fs := face.H
	bg := [3]float32{float32(p.bg[0]), float32(p.bg[1]), float32(p.bg[2])}
	out := NewTensor(1, outH, outW)
	nn.InParallel(outH, outH*outW*64, func(start, end int) {
		for oy := start; oy < end; oy++ {
			for ox := 0; ox < outW; ox++ {
				spanX, spanY := float32(winW)/s, float32(winH)/s
				x := winX + cx*float32(winW) - spanX/2 + (float32(ox)+0.5)*spanX/float32(outW) - 0.5
				y := winY + cy*float32(winH) - spanY/2 + (float32(oy)+0.5)*spanY/float32(outH) - 0.5

				sy := max((y+0.5)*(float32(lowH)/float32(h))-0.5, 0)
				sx := max((x+0.5)*(float32(lowW)/float32(w))-0.5, 0)
				y0, x0 := min(int(sy), lowH-1), min(int(sx), lowW-1)
				y1, x1 := min(y0+1, lowH-1), min(x0+1, lowW-1)
				fy, fx := sy-float32(y0), sx-float32(x0)
				field := func(t Tensor, c int) float32 {
					pl := t.Plane(c)
					top := pl[y0*lowW+x0]*(1-fx) + pl[y0*lowW+x1]*fx
					bottom := pl[y1*lowW+x0]*(1-fx) + pl[y1*lowW+x1]*fx
					return top*(1-fy) + bottom*fy
				}
				gx := field(e.grid, 0)
				gy := field(e.grid, 1)
				fw, fh := float32(w), float32(h)
				u := (2*x+1)/fw - 1 + gx
				vv := (2*y+1)/fh - 1 + gy
				tx := clamp(((u+1)*fw-1)/2, 0, fw-1)
				ty := clamp(((vv+1)*fh-1)/2, 0, fh-1)
				sx0, sy0 := int(math.Floor(float64(tx))), int(math.Floor(float64(ty)))
				gfx, gfy := tx-float32(sx0), ty-float32(sy0)
				inX, inY := sx0+1 < w, sy0+1 < h
				picture := func(c, yy, xx int) float32 {
					if yy >= fy0 && yy < fy0+fs && xx >= fx0 && xx < fx0+fs {
						return face.Plane(c)[(yy-fy0)*fs+xx-fx0]
					}
					return high.Plane(c)[yy*w+xx]
				}
				var px [4]float32
				for c := range 4 {
					v00 := picture(c, sy0, sx0)
					var v01, v10, v11 float32
					if inX {
						v01 = picture(c, sy0, sx0+1)
					}
					if inY {
						v10 = picture(c, sy0+1, sx0)
					}
					if inX && inY {
						v11 = picture(c, sy0+1, sx0+1)
					}
					warped := v00*(1-gfx)*(1-gfy) + v01*gfx*(1-gfy) + v10*(1-gfx)*gfy + v11*gfx*gfy
					al := field(e.alpha, c)
					px[c] = field(e.color, c)*al + warped*(1-al)
				}
				al := float32(unit(px[3]))
				var rgb [3]float32
				for c := range 3 {
					rgb[c] = float32(math.Floor(float64(float32(linearToSRGB(unit(px[c])))*255*al + bg[c]*(1-al) + 0.5)))
				}
				out.Data[oy*outW+ox] = rgb[0] + 256*rgb[1] + 65536*rgb[2]
			}
		}
	})
	return out
}

// SetSharpen sets how hard the mouth and the closed eyes are sharpened on
// the larger picture, where the face morpher's fields arrive blown up from
// 192 and what they paint arrives blurred. Zero, the default, leaves them as
// the enlargement made them; 0.8 is what a talking face was settled on, where
// the lash line and the lips read as drawn rather than smudged, and 1.2 is
// about as far as it is worth going. The colour cannot drift whatever the
// amount, only the light. It does nothing at a scale of one, where nothing
// was enlarged.
//
// On the card it rebuilds the passes, so it is set once, early.
func (p *Poser) SetSharpen(amount float32) error {
	if p.closed {
		return errClosed
	}
	if amount < 0 {
		return fmt.Errorf("tha3: a sharpening of %v is below zero", amount)
	}
	if amount == p.sharpen {
		return nil
	}
	old := p.sharpen
	p.sharpen = amount
	p.haveLast = false
	if p.gpu == nil {
		return nil
	}
	p.gpu.close()
	p.gpu = nil
	if err := p.useVulkan(p.gpuTrace); err != nil {
		p.sharpen = old
		return fmt.Errorf("tha3: the card for a sharpening of %v: %w", amount, err)
	}
	return nil
}

// SetHeldBody makes a pose that leaves the head, the neck and the body where
// they were reuse what the rotator and the editor decided for the last one:
// only the brows and the face run again, and the frame is made from their
// output under the warp and the repaint already there. On the card that is
// two thirds off such a pose.
//
// It is an approximation: the editor decided its repaint having seen the
// last face, so a mouth that opens wide while the head is well turned may
// show it. The warp itself follows the body, which has not moved.
//
// On the card it rebuilds the passes, so it is set once, early.
func (p *Poser) SetHeldBody(held bool) error {
	if p.closed {
		return errClosed
	}
	if held == p.heldBody {
		return nil
	}
	p.heldBody = held
	p.haveLast = false
	if p.gpu == nil {
		return nil
	}
	p.gpu.close()
	p.gpu = nil
	if err := p.useVulkan(p.gpuTrace); err != nil {
		p.heldBody = !held
		return fmt.Errorf("tha3: the card for a held body: %w", err)
	}
	return nil
}
