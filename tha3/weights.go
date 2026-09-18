package tha3

import (
	"fmt"
	"path/filepath"
	"slices"

	"github.com/ThiraSoft/golem/tensors"
)

// weights is one network's safetensors file, read by state-dict name. The
// first error sticks: a network is built by a few dozen calls in a row, and
// checking once at the end reads better than checking each.
type weights struct {
	model   *tensors.Model
	network string
	failed  error
}

func openWeights(dir, network string) (*weights, error) {
	m, err := tensors.Open(filepath.Join(dir, network+".safetensors"))
	if err != nil {
		return nil, err
	}
	return &weights{model: m, network: network}, nil
}

func (w *weights) err() error   { return w.failed }
func (w *weights) close() error { return w.model.Close() }

// get returns the named tensor as float32, after checking its shape.
func (w *weights) get(name string, shape ...int) []float32 {
	if w.failed != nil {
		return nil
	}
	t, err := w.model.Get(name)
	if err != nil {
		w.failed = fmt.Errorf("tha3: %s: %w", w.network, err)
		return nil
	}
	if !slices.Equal(t.Shape, shape) {
		w.failed = fmt.Errorf("tha3: %s: %s is %v, want %v", w.network, name, t.Shape, shape)
		return nil
	}
	data, err := t.F32()
	if err != nil {
		w.failed = fmt.Errorf("tha3: %s: %s: %w", w.network, name, err)
	}
	return data
}

func (w *weights) conv(name string, in, out, k, stride, pad, groups int, bias bool) Conv2d {
	c := Conv2d{In: in, Out: out, K: k, Stride: stride, Pad: pad, Groups: groups}
	c.Weight = w.get(name+".weight", out, in/groups, k, k)
	if bias {
		c.Bias = w.get(name+".bias", out)
	}
	return c
}

// convT is the only transposed convolution these networks hold: depthwise,
// 4×4, stride 2, padding 1, doubling the picture.
func (w *weights) convT(name string, ch int) ConvTranspose2d {
	return ConvTranspose2d{C: ch, K: 4, Stride: 2, Pad: 1, Weight: w.get(name+".weight", ch, 1, 4, 4)}
}

func (w *weights) norm(name string, ch int) *InstanceNorm {
	return &InstanceNorm{Weight: w.get(name+".weight", ch), Bias: w.get(name+".bias", ch)}
}
