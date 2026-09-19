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

func (f faceFields) resized(h, w int) faceFields {
	return faceFields{resize(f.grid, h, w), resize(f.mouthAlpha, h, w), resize(f.mouthColor, h, w), resize(f.eyeAlpha, h, w), resize(f.eyeColor, h, w)}
}
