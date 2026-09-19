package tha3

// faceMorpher is FaceMorpher09: on the 192×192 crop of the face, it moves the
// irises and the mouth by warping, repaints what the warp could not produce,
// then repaints the eyes.
type faceMorpher struct {
	body                                          *encoderDecoder
	irisMouthGrid, irisMouthColor, irisMouthAlpha block
	eyeColor, eyeAlpha                            block
}

func newFaceMorpher(w *weights) *faceMorpher {
	return &faceMorpher{
		body:           newEncoderDecoder(w, "body", 192, 4, 27, 64, 24, 6, ReLU),
		irisMouthGrid:  w.head("iris_mouth_grid_change", 64, 2, false, nil),
		irisMouthColor: w.head("iris_mouth_color_change.0", 64, 4, true, Tanh),
		irisMouthAlpha: w.head("iris_mouth_alpha.0", 64, 1, true, Sigmoid),
		eyeColor:       w.head("eye_color_change.0", 64, 4, true, Tanh),
		eyeAlpha:       w.head("eye_alpha.0", 64, 1, true, Sigmoid),
	}
}

func (m *faceMorpher) forward(image Tensor, pose []float32, t tracer) (Tensor, faceFields) {
	f := m.body.forward(image, pose, t)
	ff := faceFields{
		grid:       m.irisMouthGrid.Apply(f),
		mouthAlpha: m.irisMouthAlpha.Apply(f),
		mouthColor: m.irisMouthColor.Apply(f),
		eyeAlpha:   m.eyeAlpha.Apply(f),
		eyeColor:   m.eyeColor.Apply(f),
	}
	out := ff.apply(image)
	t.emit("out.0", out)
	return out, ff
}

// faceFields is what the face morpher decides, apart from the picture it
// applies it to: the same fields, resized, move a larger picture the same way.
type faceFields struct{ grid, mouthAlpha, mouthColor, eyeAlpha, eyeColor Tensor }

func (f faceFields) apply(image Tensor) Tensor {
	warped := applyGridChange(f.grid, image)
	mouth := applyColorChange(f.mouthAlpha, f.mouthColor, warped)
	return applyColorChange(f.eyeAlpha, f.eyeColor, mouth)
}

// applySharp is apply with the mouth and the eyes sharpened by amount, for
// the larger picture, where the fields arrive blown up from 192. An amount of
// zero is apply itself. The two masks are the morpher's own alphas, so only
// what it painted is touched: the cheeks and the nose, which come from the
// larger picture unpainted, stay as sharp as they were.
func (f faceFields) applySharp(image Tensor, amount float32) Tensor {
	if amount <= 0 {
		return f.apply(image)
	}
	// The outline of what is painted is the flank of the alpha's ramp,
	// and that ramp came up from 192 by bilinear too: it crosses from
	// nothing to everything over a couple of pixels of the larger
	// picture, which is what reads as a soft edge around the mouth. The
	// unsharp mask below cannot reach it, since it works on the frame the
	// ramp already blended. Steepening the ramp is what sharpens the
	// outline; the picture under it is sharp already.
	steep := f
	steep.mouthAlpha = steepen(f.mouthAlpha, amount)
	steep.eyeAlpha = steepen(f.eyeAlpha, amount)
	out := steep.apply(image)
	// One blur for both passes: the masks hardly overlap, so sharpening
	// the eyes against the blur of the picture whose mouth was sharpened
	// makes no visible difference and costs a resampling less.
	blurred := blurWide(out)
	out = applySharpen(f.mouthAlpha, blurred, out, amount)
	return applySharpen(f.eyeAlpha, blurred, out, amount)
}

// steepen pulls a mask away from its middle, so that it crosses from nothing
// to everything over fewer pixels: the same ramp, half as wide at an amount
// of one. It stops at a factor of four, past which the edge turns to steps.
func steepen(mask Tensor, amount float32) Tensor {
	out := NewTensor(mask.C, mask.H, mask.W)
	k := steepenBy(amount)
	for i, v := range mask.Data {
		out.Data[i] = min(max(0.5+(v-0.5)*k, 0), 1)
	}
	return out
}

// steepenBy is how far a sharpening of amount pulls a mask away from its
// middle. The card reads it too, so that both paths steepen alike.
func steepenBy(amount float32) float32 { return min(1+amount, 4) }

// blurRatio is how far blurWide goes down before coming back. A blur of a
// pixel or two, which is what the enlargement from 192 added, turned out to
// reach almost nothing: an unsharp mask only lifts what varies at the scale
// of its blur, and a mouth the morpher painted is mush over tens of pixels,
// not over two. Measured on a frame of 448, going down to a half moved 354
// pixels by a visible step, an eighth moved 1543.
const blurRatio = 8

// blurWide is the picture shrunk by blurRatio and back, bilinear both ways.
func blurWide(x Tensor) Tensor {
	return ResizeBilinear(ResizeBilinear(x, max(x.H/blurRatio, 1), max(x.W/blurRatio, 1)), x.H, x.W)
}

func (f faceFields) resized(h, w int) faceFields {
	return faceFields{resize(f.grid, h, w), resize(f.mouthAlpha, h, w), resize(f.mouthColor, h, w), resize(f.eyeAlpha, h, w), resize(f.eyeColor, h, w)}
}
