package tha3

import "fmt"

// tracer receives waypoints by name. Tests hand one in to compare each
// against the recording; everywhere else it is nil and emit does nothing.
type tracer func(name string, t Tensor)

func (t tracer) emit(name string, x Tensor) {
	if t != nil {
		t(name, x)
	}
}

// sub prefixes every name emitted through it.
func (t tracer) sub(prefix string) tracer {
	if t == nil {
		return nil
	}
	return func(name string, x Tensor) { t(prefix+"."+name, x) }
}

type layer interface{ Apply(Tensor) Tensor }

// block is what these networks put in one Sequential: convolutions, then an
// instance norm if any, then a non-linearity if any.
type block struct {
	convs []layer
	norm  *InstanceNorm
	act   func([]float32)
}

func (b block) Apply(x Tensor) Tensor {
	for _, c := range b.convs {
		x = c.Apply(x)
	}
	if b.norm != nil {
		b.norm.Apply(x)
	}
	if b.act != nil {
		b.act(x.Data)
	}
	return x
}

// sepBlock is create_separable_conv{3,7}_block and
// create_separable_downsample_block: a depthwise k×k at .0, a 1×1 at .1, the
// norm at .2 and the non-linearity.
func (w *weights) sepBlock(name string, in, out, k, stride, pad int, act func([]float32)) block {
	return block{
		convs: []layer{
			w.conv(name+".0", in, in, k, stride, pad, in, false),
			w.conv(name+".1", in, out, 1, 1, 0, 1, false),
		},
		norm: w.norm(name+".2", out),
		act:  act,
	}
}

// sepUpBlock is create_separable_upsample_block: a depthwise transposed 4×4
// at .0, then as sepBlock.
func (w *weights) sepUpBlock(name string, in, out int, act func([]float32)) block {
	return block{
		convs: []layer{w.convT(name+".0", in), w.conv(name+".1", in, out, 1, 1, 0, 1, false)},
		norm:  w.norm(name+".2", out),
		act:   act,
	}
}

// head is an output head: a dense 3×3 from the last feature map, and a
// sigmoid, a tanh or nothing. Heads wrapped in a Sequential upstream are
// named with their ".0"; the grid change heads are bare convolutions.
func (w *weights) head(name string, in, out int, bias bool, act func([]float32)) block {
	return block{convs: []layer{w.conv(name, in, out, 3, 1, 1, 1, bias)}, act: act}
}

// resnet is ResnetBlockSeparable: x plus a path of two separable 3×3, each
// followed by a norm, with the non-linearity between them only.
type resnet struct{ first, second block }

func (w *weights) resnet(name string, ch int, act func([]float32)) resnet {
	p := name + ".resnet_path"
	separable := func(at string) []layer {
		return []layer{
			w.conv(p+"."+at+".0", ch, ch, 3, 1, 1, ch, false),
			w.conv(p+"."+at+".1", ch, ch, 1, 1, 0, 1, false),
		}
	}
	return resnet{
		first:  block{convs: separable("0"), norm: w.norm(p+".1", ch), act: act},
		second: block{convs: separable("3"), norm: w.norm(p+".4", ch)},
	}
}

func (r resnet) Apply(x Tensor) Tensor {
	y := r.second.Apply(r.first.Apply(x))
	for i, v := range x.Data {
		y.Data[i] += v
	}
	return y
}

func indexed(prefix, part string, i int) string { return fmt.Sprintf("%s.%s.%d", prefix, part, i) }
