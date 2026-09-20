package tha3

// The poser on a Vulkan device.
//
// Nothing here computes. The networks the processor path built from the
// weights are walked once and described, operation for operation, on a
// vk.THA3Graph: the same convolutions with the same weights, in the same
// order, under the same waypoint names. vk plans the arena, records the image
// pass and the pose pass once, and runs them.

import (
	"fmt"
	"sort"
	"time"

	"github.com/ThiraSoft/golem/vk"
)

// gpuPoser is the card's side of a Poser.
type gpuPoser struct {
	dev         *vk.Device
	run         *vk.THA3Runner
	image, high vk.THA3Tensor // high only when the scale is above one
}

// UseVulkan moves SetImage and Pose to the first Vulkan device. It fails
// rather than falling back: a poser the caller believes is on the card and is
// not would drop to half a frame a second without a word.
func (p *Poser) UseVulkan() error { return p.useVulkan(false) }

// useVulkan with trace set keeps a copy of every waypoint the processor's
// tracer would emit, for the parity test.
func (p *Poser) useVulkan(trace bool) error {
	if p.closed {
		return errClosed
	}
	if p.gpu != nil {
		return nil
	}
	dev, err := vk.Open()
	if err != nil {
		return err
	}
	d := &describer{g: vk.NewTHA3Graph(), trace: trace}
	image, high := d.poser(p)
	run, err := d.g.Build(dev, true)
	if err != nil {
		dev.Close()
		return fmt.Errorf("tha3: %w", err)
	}
	q := &gpuPoser{dev: dev, run: run, image: image, high: high}
	// A picture already set goes to the card before the card is taken: if it
	// cannot, the poser stays on the processor, whole.
	if p.image.Data != nil {
		if err := q.setImage(p, p.image, p.high); err != nil {
			q.close()
			return err
		}
	}
	p.gpu, p.gpuTrace = q, trace
	p.haveLast = false
	return nil
}

func (q *gpuPoser) setImage(p *Poser, img, high Tensor) error {
	if err := q.run.Write(q.image, img.Data); err != nil {
		return err
	}
	if high.Data != nil {
		if err := q.run.Write(q.high, high.Data); err != nil {
			return err
		}
	}
	start := time.Now()
	if err := q.run.RunImage(); err != nil {
		return err
	}
	p.timings[netEyebrowDecomposer] = time.Since(start)
	return nil
}

func (q *gpuPoser) pose(p *Poser, pose [NumParams]float32, from stage, short bool) (Tensor, error) {
	var buf [toneAt + 3]float32
	copy(buf[:], pose[:])
	z := p.zoom.orWhole()
	buf[zoomAt], buf[zoomAt+1], buf[zoomAt+2] = float32(z.Scale), float32(z.X), float32(z.Y)
	buf[toneAt], buf[toneAt+1], buf[toneAt+2] = p.eyeTone[0], p.eyeTone[1], p.eyeTone[2]
	q.run.SetPose(buf[:])
	run := q.run.RunPoseFrom
	if short {
		run = q.run.RunPoseShort
	}
	total, err := run(int(from))
	if err != nil {
		return Tensor{}, err
	}
	// The card's clock gives each network's share; the wall clock around the
	// submission gives the total the shares are scaled to. The copy of the
	// frame out of the readback buffer comes after and is not counted.
	if short {
		// A shortened pass is not stamped: the networks it left out
		// took no time, and the others keep what they last took.
		p.timings[netRotator], p.timings[netEditor] = 0, 0
	}
	spans, err := q.run.Spans()
	if err != nil {
		return Tensor{}, err
	}
	var sum uint64
	for _, s := range spans {
		sum += s.Ticks
	}
	for _, s := range spans {
		if sum > 0 {
			p.timings[s.Label] = time.Duration(float64(total) * float64(s.Ticks) / float64(sum))
		}
	}
	c := 4
	if p.finish {
		c = 1
	}
	return Tensor{C: c, H: p.view.Dy() * p.scale, W: p.view.Dx() * p.scale, Data: q.run.Output(0)}, nil
}

func (q *gpuPoser) close() {
	q.run.Close()
	q.dev.Close()
}

// describer walks the processor's networks and describes them on a graph.
type describer struct {
	g     *vk.THA3Graph
	trace bool
}

func (d *describer) emit(name string, t vk.THA3Tensor) {
	if d.trace {
		d.g.Trace(name, t)
	}
}

func (d *describer) weights(data []float32) vk.THA3Weights { return d.g.Weights(data) }

func (d *describer) bias(b []float32) vk.THA3Weights {
	if b == nil {
		return vk.THA3Weights{}
	}
	return d.weights(b)
}

// conv is one of a body block's convolutions: depthwise, pointwise, or the
// transposed one of an upsampling block.
func (d *describer) conv(x vk.THA3Tensor, l layer) vk.THA3Tensor {
	switch c := l.(type) {
	case Conv2d:
		switch {
		case c.Groups == c.In && c.Out == c.In && c.In > 1:
			return d.g.DW(x, d.weights(c.Weight), c.K, c.Stride, c.Pad)
		case c.Groups == 1 && c.K == 1 && c.Stride == 1 && c.Pad == 0:
			return d.g.PW(x, d.weights(c.Weight), d.bias(c.Bias), c.Out)
		}
		panic(fmt.Sprintf("tha3: no card kernel for a %dx%d convolution in a body", c.K, c.K))
	case ConvTranspose2d:
		return d.g.DWT(x, d.weights(c.Weight))
	}
	panic(fmt.Sprintf("tha3: no card kernel for %T", l))
}

// block is block.Apply: the convolutions, then the norm with the block's
// non-linearity fused into it.
func (d *describer) block(x vk.THA3Tensor, b block, act vk.THA3Act) vk.THA3Tensor {
	for _, c := range b.convs {
		x = d.conv(x, c)
	}
	if b.act == nil {
		act = vk.THA3None
	}
	return d.g.Norm(x, d.weights(b.norm.Weight), d.weights(b.norm.Bias), act, nil)
}

// resnet is resnet.Apply: the second norm carries the residual add.
func (d *describer) resnet(x vk.THA3Tensor, r resnet, act vk.THA3Act) vk.THA3Tensor {
	y := d.block(x, r.first, act)
	for _, c := range r.second.convs {
		y = d.conv(y, c)
	}
	return d.g.Norm(y, d.weights(r.second.norm.Weight), d.weights(r.second.norm.Bias), vk.THA3None, &x)
}

func (d *describer) layer(x vk.THA3Tensor, l layer, act vk.THA3Act) vk.THA3Tensor {
	switch b := l.(type) {
	case block:
		return d.block(x, b, act)
	case resnet:
		return d.resnet(x, b, act)
	}
	panic(fmt.Sprintf("tha3: no card description for %T", l))
}

// head is an output head; its activation is named by the caller, which knows
// which head it is.
func (d *describer) head(x vk.THA3Tensor, b block, act vk.THA3Act) vk.THA3Tensor {
	c := b.convs[0].(Conv2d)
	return d.g.Head(x, d.weights(c.Weight), d.bias(c.Bias), c.Out, act)
}

func (d *describer) encoderDecoder(net string, e *encoderDecoder, x vk.THA3Tensor, from, n int, act vk.THA3Act) vk.THA3Tensor {
	for i, b := range e.down {
		x = d.block(x, b, act)
		d.emit(net+"."+indexed(e.prefix, "downsample_blocks", i), x)
	}
	if n > 0 {
		x = d.g.Concat(x, d.g.PoseSlice(from, n, x.H, x.W))
	}
	for i, l := range e.bottleneck {
		x = d.layer(x, l, act)
		d.emit(net+"."+indexed(e.prefix, "bottleneck_blocks", i), x)
	}
	for i, b := range e.up {
		x = d.block(x, b, act)
		d.emit(net+"."+indexed(e.prefix, "upsample_blocks", i), x)
	}
	return x
}

func (d *describer) resizeEncoderDecoder(net string, e *resizeEncoderDecoder, x vk.THA3Tensor, act vk.THA3Act) vk.THA3Tensor {
	for i, b := range e.down {
		x = d.block(x, b, act)
		d.emit(net+"."+indexed(e.prefix, "downsample_blocks", i), x)
	}
	for i, r := range e.bottleneck {
		x = d.resnet(x, r, act)
		d.emit(net+"."+indexed(e.prefix, "bottleneck_blocks", i), x)
	}
	for i, b := range e.up {
		x = d.block(d.g.Upsample2(x), b, act)
		d.emit(net+"."+indexed(e.prefix, "upsample_blocks", i), x)
	}
	return x
}

func (d *describer) unet(net string, u *unet, x vk.THA3Tensor, act vk.THA3Act) vk.THA3Tensor {
	features := make([]vk.THA3Tensor, 0, len(u.down))
	for i, b := range u.down {
		x = d.block(x, b, act)
		features = append(features, x)
		d.emit(net+"."+indexed(u.prefix, "downsample_blocks", i), x)
	}
	for i, r := range u.bottleneck {
		x = d.resnet(x, r, act)
		d.emit(net+"."+indexed(u.prefix, "bottleneck_blocks", i), x)
	}
	for i, b := range u.up {
		x = d.block(d.g.Concat(d.g.Upsample2(x), features[len(features)-i-2]), b, act)
		d.emit(net+"."+indexed(u.prefix, "upsample_blocks", i), x)
	}
	return x
}

// cardFields is what a network decided, on the card, by the name of the
// field on the processor: for the larger picture, as the fields types there.
type cardFields map[string]vk.THA3Tensor

func (d *describer) decomposer(m *eyebrowDecomposer, image vk.THA3Tensor) (eyebrow, background vk.THA3Tensor, fields cardFields) {
	const net = netEyebrowDecomposer
	f := d.encoderDecoder(net, m.body, image, 0, 0, vk.THA3ReLU)
	fields = cardFields{
		"backgroundAlpha": d.head(f, m.backgroundAlpha, vk.THA3Sigmoid),
		"backgroundColor": d.head(f, m.backgroundColor, vk.THA3Tanh),
		"eyebrowAlpha":    d.head(f, m.eyebrowAlpha, vk.THA3Sigmoid),
		"eyebrowColor":    d.head(f, m.eyebrowColor, vk.THA3Tanh),
	}
	eyebrow, background = d.decompose(fields, image)
	d.emit(net+".out.0", eyebrow)
	d.emit(net+".out.3", background)
	return eyebrow, background, fields
}

// decompose is decomposerFields.apply.
func (d *describer) decompose(f cardFields, image vk.THA3Tensor) (eyebrow, background vk.THA3Tensor) {
	background = d.g.ColorChange(f["backgroundAlpha"], f["backgroundColor"], image)
	eyebrow = d.g.ColorChange(f["eyebrowAlpha"], image, f["eyebrowColor"])
	return eyebrow, background
}

func (d *describer) combiner(m *eyebrowCombiner, background, eyebrow vk.THA3Tensor) (vk.THA3Tensor, cardFields) {
	const net = netEyebrowCombiner
	f := d.encoderDecoder(net, m.body, d.g.Concat(background, eyebrow), 0, eyebrowParams, vk.THA3ReLU)
	fields := cardFields{
		"grid":  d.head(f, m.gridChange, vk.THA3None),
		"alpha": d.head(f, m.alpha, vk.THA3Sigmoid),
		"color": d.head(f, m.color, vk.THA3Tanh),
	}
	out := d.combine(fields, background, eyebrow)
	d.emit(net+".out.2", out)
	return out, fields
}

// combine is combinerFields.apply.
func (d *describer) combine(f cardFields, background, eyebrow vk.THA3Tensor) vk.THA3Tensor {
	morphed := d.g.ColorChange(f["alpha"], f["color"], d.g.Warp(eyebrow, f["grid"]))
	return d.g.RGBHalfAlpha(morphed, background)
}

// face is faceMorpher.forward, with the eyes' colour toned by the gain the
// pose buffer carries: the pass is recorded once whatever the picture, and
// the gain arrives with the pose, as the zoom does.
func (d *describer) face(m *faceMorpher, image vk.THA3Tensor) (vk.THA3Tensor, cardFields) {
	const net = netFaceMorpher
	f := d.encoderDecoder(net, m.body, image, eyebrowParams, faceParamsEnd-eyebrowParams, vk.THA3ReLU)
	fields := cardFields{
		"grid":       d.head(f, m.irisMouthGrid, vk.THA3None),
		"mouthAlpha": d.head(f, m.irisMouthAlpha, vk.THA3Sigmoid),
		"mouthColor": d.head(f, m.irisMouthColor, vk.THA3Tanh),
		"eyeAlpha":   d.head(f, m.eyeAlpha, vk.THA3Sigmoid),
		"eyeColor":   d.g.Tone(d.head(f, m.eyeColor, vk.THA3Tanh), toneAt),
	}
	out := d.morph(fields, image)
	d.emit(net+".out.0", out)
	return out, fields
}

// morph is faceFields.apply.
func (d *describer) morph(f cardFields, image vk.THA3Tensor) vk.THA3Tensor {
	mouth := d.g.ColorChange(f["mouthAlpha"], f["mouthColor"], d.g.Warp(image, f["grid"]))
	return d.g.ColorChange(f["eyeAlpha"], f["eyeColor"], mouth)
}

// steepened is the fields with their two alphas pulled away from the middle,
// as faceFields.applySharp steepens them.
func (d *describer) steepened(f cardFields, amount float32) cardFields {
	if amount <= 0 {
		return f
	}
	out := cardFields{}
	for name, t := range f {
		out[name] = t
	}
	by := steepenBy(amount)
	out["mouthAlpha"] = d.g.Steepen(f["mouthAlpha"], by)
	out["eyeAlpha"] = d.g.Steepen(f["eyeAlpha"], by)
	return out
}

// sharpen is faceFields.applySharp's second half: the mouth and the eyes of
// out brought back to an edge, each under the morpher's own alpha.
func (d *describer) sharpen(f cardFields, out vk.THA3Tensor, amount float32) vk.THA3Tensor {
	if amount <= 0 {
		return out
	}
	blurred := d.g.Resize(d.g.Resize(out, max(out.H/blurRatio, 1), max(out.W/blurRatio, 1)), out.H, out.W)
	out = d.g.Sharpen(f["mouthAlpha"], blurred, out, amount)
	return d.g.Sharpen(f["eyeAlpha"], blurred, out, amount)
}

// resized is every field resized to h×w, for the larger picture.
func (d *describer) resized(f cardFields, h, w int) cardFields {
	// In the order of the names, so that the graph is the same every time.
	names := make([]string, 0, len(f))
	for name := range f {
		names = append(names, name)
	}
	sort.Strings(names)
	out := cardFields{}
	for _, name := range names {
		out[name] = d.g.Resize(f[name], h, w)
	}
	return out
}

func (d *describer) rotator(m *rotator, image vk.THA3Tensor) (warped, grid vk.THA3Tensor) {
	const net = netRotator
	in := d.g.Concat(image, d.g.PoseSlice(faceParamsEnd, NumParams-faceParamsEnd, image.H, image.W))
	f := d.resizeEncoderDecoder(net, m.body, in, vk.THA3Leaky)
	grid = d.head(f, m.gridChange, vk.THA3None)
	warped = d.g.Warp(image, grid)
	d.emit(net+".out.1", warped)
	d.emit(net+".out.2", grid)
	return warped, grid
}

// editor returns the frame, or with high set only the editor's fields: the
// frame is then made from them on the larger picture, and the one at 512
// would be thrown away.
func (d *describer) editor(m *editor, original, warped, grid vk.THA3Tensor, high bool) (vk.THA3Tensor, cardFields) {
	const net = netEditor
	in := d.g.Concat(original, warped, grid, d.g.PoseSlice(faceParamsEnd, NumParams-faceParamsEnd, original.H, original.W))
	f := d.unet(net, m.body, in, vk.THA3Leaky)
	fields := cardFields{
		"grid":  d.head(f, m.gridChange, vk.THA3None),
		"alpha": d.head(f, m.alpha, vk.THA3Sigmoid),
		"color": d.head(f, m.color, vk.THA3Tanh),
	}
	if high {
		return vk.THA3Tensor{}, fields
	}
	rewarped := d.g.WarpAdd(original, fields["grid"], grid)
	out := d.g.ColorChange(fields["alpha"], fields["color"], rewarped)
	d.emit(net+".out.0", out)
	return out, fields
}

// poser describes SetImage and Pose as Poser runs them on the processor and
// returns the picture's input tensor, and the larger picture's when the
// scale is above one.
func (d *describer) poser(p *Poser) (image, high vk.THA3Tensor) {
	g := d.g
	k := p.scale
	image = g.Input(4, Size, Size)
	if k > 1 {
		high = g.Input(4, Size*k, Size*k)
	}

	g.SetPhase(vk.THA3ImagePhase)
	crop := g.Crop(image, 64, 192, 128, 128)
	d.emit(netEyebrowDecomposer+".in.0", crop)
	eyebrow, background, decomposed := d.decomposer(p.decomposer, crop)
	var hiEyebrow, hiBackground vk.THA3Tensor
	if k > 1 {
		hiEyebrow, hiBackground = d.decompose(d.resized(decomposed, 128*k, 128*k), g.Crop(high, 64*k, 192*k, 128*k, 128*k))
	}

	g.SetPhase(vk.THA3PosePhase)
	d.emit(netEyebrowCombiner+".in.0", background)
	d.emit(netEyebrowCombiner+".in.1", eyebrow)
	eyebrows, combined := d.combiner(p.combiner, background, eyebrow)
	var hiEyebrows vk.THA3Tensor
	if k > 1 {
		hiEyebrows = d.combine(d.resized(combined, 128*k, 128*k), hiBackground, hiEyebrow)
	}
	g.Stamp(netEyebrowCombiner)

	if g.Entry() != int(stageFace) {
		panic("tha3: the face's entry is not the face's stage")
	}

	faceIn := g.Paste(g.Crop(image, 32, 160, 192, 192), eyebrows, 32, 32)
	d.emit(netFaceMorpher+".in.0", faceIn)
	face, faced := d.face(p.face, faceIn)
	var hiFace vk.THA3Tensor
	if k > 1 {
		hiFaceIn := g.Paste(g.Crop(high, 32*k, 160*k, 192*k, 192*k), hiEyebrows, 32*k, 32*k)
		hiFaced := d.resized(faced, 192*k, 192*k)
		// The morph composites through the steepened alphas, which is
		// what sharpens the outline of the mouth; the unsharp mask
		// after it is weighed by the alphas as they came, so that it
		// reaches everything the morpher painted.
		hiFace = d.sharpen(hiFaced, d.morph(d.steepened(hiFaced, p.sharpen), hiFaceIn), p.sharpen)
	}
	g.Stamp(netFaceMorpher)

	if g.Entry() != int(stageBody) {
		panic("tha3: the body's entry is not the body's stage")
	}

	g.Skip()
	full := g.Paste(image, face, 32, 160)
	half := g.Resize(full, Size/2, Size/2)
	d.emit(netRotator+".in.0", half)
	warped, grid := d.rotator(p.rotator, half)
	g.Stamp(netRotator)

	warped = g.Resize(warped, Size, Size)
	grid = g.Resize(grid, Size, Size)
	d.emit(netEditor+".in.0", full)
	d.emit(netEditor+".in.1", warped)
	d.emit(netEditor+".in.2", grid)
	if k == 1 {
		// Finishing at the scale of one: the larger picture is the
		// picture, and the face on it the face.
		high, hiFace = image, face
	}
	frame, edited := d.editor(p.editor, full, warped, grid, k > 1 || p.finish)
	if p.heldBody {
		// The face alone may move without the rotator and the editor
		// running again: the frame is made from what they left.
		g.SkipEnd()
	}
	v := p.view
	switch {
	case p.finish:
		frame = g.EditHighRGB(high, hiFace, 32*k, 160*k, edited["grid"], grid, edited["alpha"], edited["color"],
			v.Min.Y*k, v.Min.X*k, v.Dy()*k, v.Dx()*k, p.bg, zoomAt, v.Dy()*k, v.Dx()*k)
	case k > 1:
		frame = g.EditHigh(high, hiFace, 32*k, 160*k, edited["grid"], grid, edited["alpha"], edited["color"],
			v.Min.Y*k, v.Min.X*k, v.Dy()*k, v.Dx()*k)
	case v != whole:
		frame = g.Crop(frame, v.Min.Y, v.Min.X, v.Dy(), v.Dx())
	}
	g.Output(frame)
	g.Stamp(netEditor)
	if k == 1 {
		high = vk.THA3Tensor{}
	}
	return image, high
}
