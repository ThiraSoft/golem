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
func (d *eyebrowDecomposer) forward(image Tensor, t tracer) (eyebrow, background Tensor, fields decomposerFields) {
	f := d.body.forward(image, nil, t)
	fields = decomposerFields{
		backgroundAlpha: d.backgroundAlpha.Apply(f),
		backgroundColor: d.backgroundColor.Apply(f),
		eyebrowAlpha:    d.eyebrowAlpha.Apply(f),
		eyebrowColor:    d.eyebrowColor.Apply(f),
	}
	eyebrow, background = fields.apply(image)
	t.emit("out.0", eyebrow)
	t.emit("out.3", background)
	return eyebrow, background, fields
}

// decomposerFields is what the decomposer decides, apart from the picture.
type decomposerFields struct{ backgroundAlpha, backgroundColor, eyebrowAlpha, eyebrowColor Tensor }

func (f decomposerFields) apply(image Tensor) (eyebrow, background Tensor) {
	background = applyColorChange(f.backgroundAlpha, f.backgroundColor, image)
	eyebrow = applyColorChange(f.eyebrowAlpha, image, f.eyebrowColor)
	return eyebrow, background
}

func (f decomposerFields) resized(h, w int) decomposerFields {
	return decomposerFields{resize(f.backgroundAlpha, h, w), resize(f.backgroundColor, h, w), resize(f.eyebrowAlpha, h, w), resize(f.eyebrowColor, h, w)}
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
func (c *eyebrowCombiner) forward(background, eyebrow Tensor, pose []float32, t tracer) (Tensor, combinerFields) {
	f := c.body.forward(Concat(background, eyebrow), pose, t)
	fields := combinerFields{grid: c.gridChange.Apply(f), alpha: c.alpha.Apply(f), color: c.color.Apply(f)}
	out := fields.apply(background, eyebrow)
	t.emit("out.2", out)
	return out, fields
}

// combinerFields is what the combiner decides, apart from the layers.
type combinerFields struct{ grid, alpha, color Tensor }

func (f combinerFields) apply(background, eyebrow Tensor) Tensor {
	warped := applyGridChange(f.grid, eyebrow)
	morphed := applyColorChange(f.alpha, f.color, warped)
	alpha := morphed.Channels(3, 4).Clone()
	for i, v := range alpha.Data {
		alpha.Data[i] = (v + 1) / 2
	}
	return applyRGBChange(alpha, morphed, background)
}

func (f combinerFields) resized(h, w int) combinerFields {
	return combinerFields{resize(f.grid, h, w), resize(f.alpha, h, w), resize(f.color, h, w)}
}
