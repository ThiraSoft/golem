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

func (e *editor) forward(original, warped, gridChange Tensor, pose []float32, t tracer) Tensor {
	in := Concat(original, warped, gridChange, Broadcast(pose, original.H, original.W))
	f := e.body.forward(in, t)
	grid := e.gridChange.Apply(f)
	for i, v := range gridChange.Data {
		grid.Data[i] += v
	}
	rewarped := applyGridChange(grid, original)
	out := applyColorChange(e.alpha.Apply(f), e.color.Apply(f), rewarped)
	t.emit("out.0", out)
	return out
}
