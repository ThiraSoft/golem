package tha3

// editor is Editor07 at 512×512: it refines the rotator's coarse warp, which
// was computed at half resolution, and repaints what warping cannot give.
type editor struct {
	body                     *unet
	color, alpha, gridChange block
}

func newEditor(w *weights) *editor {
	return &editor{
		body:       newUNet(w, "body", 512, 16, 32, 64, 6, leaky),
		color:      w.head("color_change_creator.0", 32, 4, true, Tanh),
		alpha:      w.head("alpha_creator.0", 32, 4, true, Sigmoid),
		gridChange: w.head("grid_change_creator", 32, 2, false, nil),
	}
}

func (e *editor) forward(original, warped, gridChange Tensor, pose []float32, t tracer) (Tensor, editorFields) {
	in := Concat(original, warped, gridChange, Broadcast(pose, original.H, original.W))
	f := e.body.forward(in, t)
	grid := e.gridChange.Apply(f)
	for i, v := range gridChange.Data {
		grid.Data[i] += v
	}
	fields := editorFields{grid: grid, alpha: e.alpha.Apply(f), color: e.color.Apply(f)}
	out := fields.apply(original)
	t.emit("out.0", out)
	return out, fields
}

// editorFields is what the editor decides, apart from the picture: the whole
// warp, the rotator's included, and the repaint over it.
type editorFields struct{ grid, alpha, color Tensor }

func (f editorFields) apply(original Tensor) Tensor {
	return applyColorChange(f.alpha, f.color, applyGridChange(f.grid, original))
}

func (f editorFields) resized(h, w int) editorFields {
	return editorFields{resize(f.grid, h, w), resize(f.alpha, h, w), resize(f.color, h, w)}
}
