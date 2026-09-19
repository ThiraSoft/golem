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

func (m *faceMorpher) forward(image Tensor, pose []float32, t tracer) Tensor {
	f := m.body.forward(image, pose, t)
	warped := applyGridChange(m.irisMouthGrid.Apply(f), image)
	mouth := applyColorChange(m.irisMouthAlpha.Apply(f), m.irisMouthColor.Apply(f), warped)
	out := applyColorChange(m.eyeAlpha.Apply(f), m.eyeColor.Apply(f), mouth)
	t.emit("out.0", out)
	return out
}
