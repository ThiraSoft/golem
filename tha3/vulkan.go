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
	"time"

	"github.com/ThiraSoft/golem/vk"
)

// gpuPoser is the card's side of a Poser.
type gpuPoser struct {
	dev   *vk.Device
	run   *vk.THA3Runner
	image vk.THA3Tensor
}

// UseVulkan moves SetImage and Pose to the first Vulkan device. It fails
// rather than falling back: a poser the caller believes is on the card and is
// not would drop to half a frame a second without a word.
func (p *Poser) UseVulkan() error { return p.useVulkan(false) }

// useVulkan with trace set keeps a copy of every waypoint the processor's
// tracer would emit, for the parity test.
func (p *Poser) useVulkan(trace bool) error {
	if p.gpu != nil {
		return nil
	}
	dev, err := vk.Open()
	if err != nil {
		return err
	}
	d := &describer{g: vk.NewTHA3Graph(), trace: trace}
	image := d.poser(p)
	run, err := d.g.Build(dev, true)
	if err != nil {
		dev.Close()
		return fmt.Errorf("tha3: %w", err)
	}
	p.gpu = &gpuPoser{dev: dev, run: run, image: image}
	if p.image.Data != nil {
		return p.gpu.setImage(p, p.image)
	}
	return nil
}

func (q *gpuPoser) setImage(p *Poser, img Tensor) error {
	if err := q.run.Write(q.image, img.Data); err != nil {
		return err
	}
	start := time.Now()
	if err := q.run.RunImage(); err != nil {
		return err
	}
	p.timings[netEyebrowDecomposer] = time.Since(start)
	return nil
}

func (q *gpuPoser) pose(p *Poser, pose [NumParams]float32) (Tensor, error) {
	q.run.SetPose(pose[:])
	total, err := q.run.RunPose()
	if err != nil {
		return Tensor{}, err
	}
	// The card's clock gives each network's share; the wall clock around the
	// submission gives the total the shares are scaled to, readback included.
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
	return Tensor{C: 4, H: Size, W: Size, Data: q.run.Output(0)}, nil
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

func (d *describer) decomposer(m *eyebrowDecomposer, image vk.THA3Tensor) (eyebrow, background vk.THA3Tensor) {
	const net = netEyebrowDecomposer
	f := d.encoderDecoder(net, m.body, image, 0, 0, vk.THA3ReLU)
	background = d.g.ColorChange(d.head(f, m.backgroundAlpha, vk.THA3Sigmoid), d.head(f, m.backgroundColor, vk.THA3Tanh), image)
	eyebrow = d.g.ColorChange(d.head(f, m.eyebrowAlpha, vk.THA3Sigmoid), image, d.head(f, m.eyebrowColor, vk.THA3Tanh))
	d.emit(net+".out.0", eyebrow)
	d.emit(net+".out.3", background)
	return eyebrow, background
}

func (d *describer) combiner(m *eyebrowCombiner, background, eyebrow vk.THA3Tensor) vk.THA3Tensor {
	const net = netEyebrowCombiner
	f := d.encoderDecoder(net, m.body, d.g.Concat(background, eyebrow), 0, eyebrowParams, vk.THA3ReLU)
	warped := d.g.Warp(eyebrow, d.head(f, m.gridChange, vk.THA3None))
	morphed := d.g.ColorChange(d.head(f, m.alpha, vk.THA3Sigmoid), d.head(f, m.color, vk.THA3Tanh), warped)
	out := d.g.RGBHalfAlpha(morphed, background)
	d.emit(net+".out.2", out)
	return out
}

func (d *describer) face(m *faceMorpher, image vk.THA3Tensor) vk.THA3Tensor {
	const net = netFaceMorpher
	f := d.encoderDecoder(net, m.body, image, eyebrowParams, faceParamsEnd-eyebrowParams, vk.THA3ReLU)
	warped := d.g.Warp(image, d.head(f, m.irisMouthGrid, vk.THA3None))
	mouth := d.g.ColorChange(d.head(f, m.irisMouthAlpha, vk.THA3Sigmoid), d.head(f, m.irisMouthColor, vk.THA3Tanh), warped)
	out := d.g.ColorChange(d.head(f, m.eyeAlpha, vk.THA3Sigmoid), d.head(f, m.eyeColor, vk.THA3Tanh), mouth)
	d.emit(net+".out.0", out)
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

func (d *describer) editor(m *editor, original, warped, grid vk.THA3Tensor) vk.THA3Tensor {
	const net = netEditor
	in := d.g.Concat(original, warped, grid, d.g.PoseSlice(faceParamsEnd, NumParams-faceParamsEnd, original.H, original.W))
	f := d.unet(net, m.body, in, vk.THA3Leaky)
	rewarped := d.g.WarpAdd(original, d.head(f, m.gridChange, vk.THA3None), grid)
	out := d.g.ColorChange(d.head(f, m.alpha, vk.THA3Sigmoid), d.head(f, m.color, vk.THA3Tanh), rewarped)
	d.emit(net+".out.0", out)
	return out
}

// poser describes SetImage and Pose as Poser runs them on the processor and
// returns the picture's input tensor.
func (d *describer) poser(p *Poser) vk.THA3Tensor {
	g := d.g
	image := g.Input(4, Size, Size)

	g.SetPhase(vk.THA3ImagePhase)
	crop := g.Crop(image, 64, 192, 128, 128)
	d.emit(netEyebrowDecomposer+".in.0", crop)
	eyebrow, background := d.decomposer(p.decomposer, crop)

	g.SetPhase(vk.THA3PosePhase)
	d.emit(netEyebrowCombiner+".in.0", background)
	d.emit(netEyebrowCombiner+".in.1", eyebrow)
	eyebrows := d.combiner(p.combiner, background, eyebrow)
	g.Stamp(netEyebrowCombiner)

	faceIn := g.Paste(g.Crop(image, 32, 160, 192, 192), eyebrows, 32, 32)
	d.emit(netFaceMorpher+".in.0", faceIn)
	face := d.face(p.face, faceIn)
	g.Stamp(netFaceMorpher)

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
	frame := d.editor(p.editor, full, warped, grid)
	g.Output(frame)
	g.Stamp(netEditor)
	return image
}
