package tha3

// rotator is TwoAlgoFaceBodyRotator05 at 256×256: it turns the head and the
// body by a warp of the picture. Its direct image, output 0 upstream, is not
// used by the poser and is not computed.
type rotator struct {
	body       *resizeEncoderDecoder
	gridChange block
}

func newRotator(w *weights) *rotator {
	return &rotator{
		body:       newResizeEncoderDecoder(w, "encoder_decoder", 256, 10, 64, 32, 6, leaky),
		gridChange: w.head("grid_change_creator", 64, 2, false, nil),
	}
}

func (r *rotator) forward(image Tensor, pose []float32, t tracer) (warped, gridChange Tensor) {
	f := r.body.forward(Concat(image, Broadcast(pose, image.H, image.W)), t)
	gridChange = r.gridChange.Apply(f)
	warped = applyGridChange(gridChange, image)
	t.emit("out.1", warped)
	t.emit("out.2", gridChange)
	return warped, gridChange
}
