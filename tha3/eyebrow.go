package tha3

// eyebrowDecomposer is EyebrowDecomposer03: it splits the 128×128 crop around
// the eyebrows into an eyebrow layer and the face behind it.
type eyebrowDecomposer struct {
	body                             *encoderDecoder
	backgroundAlpha, backgroundColor block
	eyebrowAlpha, eyebrowColor       block
}

func newEyebrowDecomposer(w *weights) *eyebrowDecomposer {
	return &eyebrowDecomposer{
		body:            newEncoderDecoder(w, "body", 128, 4, 0, 64, 16, 6, ReLU),
		backgroundAlpha: w.head("background_layer_alpha.0", 64, 1, true, Sigmoid),
		backgroundColor: w.head("background_layer_color_change.0", 64, 4, true, Tanh),
		eyebrowAlpha:    w.head("eyebrow_layer_alpha.0", 64, 1, true, Sigmoid),
		eyebrowColor:    w.head("eyebrow_layer_color_change.0", 64, 4, true, Tanh),
	}
}

// forward returns upstream's outputs 0 and 3. Note the eyebrow layer blends
// the other way round from the background: the picture is what the alpha
// selects, and the colour change what shows through.
func (d *eyebrowDecomposer) forward(image Tensor, t tracer) (eyebrow, background Tensor) {
	f := d.body.forward(image, nil, t)
	background = applyColorChange(d.backgroundAlpha.Apply(f), d.backgroundColor.Apply(f), image)
	eyebrow = applyColorChange(d.eyebrowAlpha.Apply(f), image, d.eyebrowColor.Apply(f))
	t.emit("out.0", eyebrow)
	t.emit("out.3", background)
	return eyebrow, background
}

// eyebrowCombiner is EyebrowMorphingCombiner03: it moves the eyebrow layer as
// the twelve eyebrow parameters say and lays it back over the face.
type eyebrowCombiner struct {
	body                     *encoderDecoder
	gridChange, alpha, color block
}

func newEyebrowCombiner(w *weights) *eyebrowCombiner {
	return &eyebrowCombiner{
		body:       newEncoderDecoder(w, "body", 128, 8, 12, 64, 16, 6, ReLU),
		gridChange: w.head("morphed_eyebrow_layer_grid_change", 64, 2, false, nil),
		alpha:      w.head("morphed_eyebrow_layer_alpha.0", 64, 1, true, Sigmoid),
		color:      w.head("morphed_eyebrow_layer_color_change.0", 64, 4, true, Tanh),
	}
}

// forward returns upstream's output 2, the one the poser uses: the morphed
// eyebrow laid over the background by its own alpha. combine_alpha feeds
// output 0 only, so it is not computed.
func (c *eyebrowCombiner) forward(background, eyebrow Tensor, pose []float32, t tracer) Tensor {
	f := c.body.forward(Concat(background, eyebrow), pose, t)
	warped := applyGridChange(c.gridChange.Apply(f), eyebrow)
	morphed := applyColorChange(c.alpha.Apply(f), c.color.Apply(f), warped)
	alpha := morphed.Channels(3, 4).Clone()
	for i, v := range alpha.Data {
		alpha.Data[i] = (v + 1) / 2
	}
	out := applyRGBChange(alpha, morphed, background)
	t.emit("out.2", out)
	return out
}
