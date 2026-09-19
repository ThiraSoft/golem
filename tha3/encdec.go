package tha3

// maxChannels caps the width of every body in the five networks.
const maxChannels = 512

// encoderDecoder is PoserEncoderDecoder00Separable, the body of the three
// face networks: a separable 3×3, halvings down to the bottleneck, the pose
// concatenated there, one plain block and resnets, doublings back up. Its
// output is the last doubling's.
type encoderDecoder struct {
	prefix     string
	down, up   []block
	bottleneck []layer
}

func newEncoderDecoder(w *weights, prefix string, size, inCh, poseParams, start, bottleneckSize, blocks int, act func([]float32)) *encoderDecoder {
	width := func(s int) int { return min(start*(size/s), maxChannels) }
	e := &encoderDecoder{prefix: prefix}
	e.down = append(e.down, w.sepBlock(indexed(prefix, "downsample_blocks", 0), inCh, start, 3, 1, 1, act))
	s, ch := size, start
	for s > bottleneckSize {
		next := width(s / 2)
		e.down = append(e.down, w.sepBlock(indexed(prefix, "downsample_blocks", len(e.down)), ch, next, 4, 2, 1, act))
		s, ch = s/2, next
	}
	e.bottleneck = append(e.bottleneck, w.sepBlock(indexed(prefix, "bottleneck_blocks", 0), ch+poseParams, ch, 3, 1, 1, act))
	for i := 1; i < blocks; i++ {
		e.bottleneck = append(e.bottleneck, w.resnet(indexed(prefix, "bottleneck_blocks", i), ch, act))
	}
	for s < size {
		next := width(s * 2)
		e.up = append(e.up, w.sepUpBlock(indexed(prefix, "upsample_blocks", len(e.up)), ch, next, act))
		s, ch = s*2, next
	}
	return e
}

func (e *encoderDecoder) forward(x Tensor, pose []float32, t tracer) Tensor {
	for i, b := range e.down {
		x = b.Apply(x)
		t.emit(indexed(e.prefix, "downsample_blocks", i), x)
	}
	if len(pose) > 0 {
		x = Concat(x, Broadcast(pose, x.H, x.W))
	}
	for i, b := range e.bottleneck {
		x = b.Apply(x)
		t.emit(indexed(e.prefix, "bottleneck_blocks", i), x)
	}
	for i, b := range e.up {
		x = b.Apply(x)
		t.emit(indexed(e.prefix, "upsample_blocks", i), x)
	}
	return x
}

// resizeEncoderDecoder is ResizeConvEncoderDecoder with separable
// convolutions, the rotator's body: a separable 7×7, halvings, resnets only
// at the bottleneck, then nearest doublings each followed by a separable 3×3
// block (named .1 inside upsample_blocks.i, after the Upsample at .0).
type resizeEncoderDecoder struct {
	prefix     string
	down, up   []block
	bottleneck []resnet
}

func newResizeEncoderDecoder(w *weights, prefix string, size, inCh, start, bottleneckSize, blocks int, act func([]float32)) *resizeEncoderDecoder {
	width := func(s int) int { return min(start*(size/s), maxChannels) }
	e := &resizeEncoderDecoder{prefix: prefix}
	e.down = append(e.down, w.sepBlock(indexed(prefix, "downsample_blocks", 0), inCh, start, 7, 1, 3, act))
	s, ch := size, start
	for s > bottleneckSize {
		next := width(s / 2)
		e.down = append(e.down, w.sepBlock(indexed(prefix, "downsample_blocks", len(e.down)), ch, next, 4, 2, 1, act))
		s, ch = s/2, next
	}
	for i := 0; i < blocks; i++ {
		e.bottleneck = append(e.bottleneck, w.resnet(indexed(prefix, "bottleneck_blocks", i), ch, act))
	}
	for s < size {
		next := width(s * 2)
		e.up = append(e.up, w.sepBlock(indexed(prefix, "upsample_blocks", len(e.up))+".1", ch, next, 3, 1, 1, act))
		s, ch = s*2, next
	}
	return e
}

func (e *resizeEncoderDecoder) forward(x Tensor, t tracer) Tensor {
	for i, b := range e.down {
		x = b.Apply(x)
		t.emit(indexed(e.prefix, "downsample_blocks", i), x)
	}
	for i, b := range e.bottleneck {
		x = b.Apply(x)
		t.emit(indexed(e.prefix, "bottleneck_blocks", i), x)
	}
	for i, b := range e.up {
		x = b.Apply(UpsampleNearest2(x))
		t.emit(indexed(e.prefix, "upsample_blocks", i), x)
	}
	return x
}

// unet is ResizeConvUNet with separable convolutions, the editor's body.
// Widths double on the way down (capped), and each step up doubles the
// picture by nearest, concatenates the feature map of the same size from the
// way down, and runs a separable 3×3 block.
type unet struct {
	prefix     string
	down, up   []block
	bottleneck []resnet
}

func newUNet(w *weights, prefix string, size, inCh, start, bottleneckSize, blocks int, act func([]float32)) *unet {
	u := &unet{prefix: prefix}
	u.down = append(u.down, w.sepBlock(indexed(prefix, "downsample_blocks", 0), inCh, start, 3, 1, 1, act))
	widths := map[int]int{size: start}
	s, ch := size, start
	for s > bottleneckSize {
		next := min(maxChannels, ch*2)
		u.down = append(u.down, w.sepBlock(indexed(prefix, "downsample_blocks", len(u.down)), ch, next, 4, 2, 1, act))
		s, ch = s/2, next
		widths[s] = ch
	}
	for i := 0; i < blocks; i++ {
		u.bottleneck = append(u.bottleneck, w.resnet(indexed(prefix, "bottleneck_blocks", i), ch, act))
	}
	for s < size {
		next := widths[s*2]
		u.up = append(u.up, w.sepBlock(indexed(prefix, "upsample_blocks", len(u.up)), ch+next, next, 3, 1, 1, act))
		s, ch = s*2, next
	}
	return u
}

func (u *unet) forward(x Tensor, t tracer) Tensor {
	features := make([]Tensor, 0, len(u.down))
	for i, b := range u.down {
		x = b.Apply(x)
		features = append(features, x)
		t.emit(indexed(u.prefix, "downsample_blocks", i), x)
	}
	for i, b := range u.bottleneck {
		x = b.Apply(x)
		t.emit(indexed(u.prefix, "bottleneck_blocks", i), x)
	}
	for i, b := range u.up {
		x = b.Apply(Concat(UpsampleNearest2(x), features[len(features)-i-2]))
		t.emit(indexed(u.prefix, "upsample_blocks", i), x)
	}
	return x
}
