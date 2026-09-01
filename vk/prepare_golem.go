package vk

// The activation side of the Golem scheme on the card.
//
// A Golem matrix is stored as A·(q ⊙ W): its columns scaled by the site's
// vector and then rotated. The product is only the right one if the activation
// meets the reciprocal of that scale and the same rotation, and neither belongs
// to the weights — one vector serves every matrix of a site, and the rotation
// is the same for all of them. So it is done once, here, to the activation, and
// the weights carry nothing about it.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/prepare_golem.comp -o shaders/prepare_golem.spv

//go:embed shaders/prepare_golem.spv
var prepareGolemSPIRV []byte

//go:generate glslc -O -DFROMQ8 --target-env=vulkan1.1 -fshader-stage=compute shaders/prepare_golem.comp -o shaders/prepare_golem_q8.spv

// prepareGolemQ8SPIRV is the same transform reading a Q8_0 activation instead
// of
// a float one. The attention's mix arrives that way — every other kernel in the
// engine reads activations in that form — and dequantizing it inside this pass
// is one multiply where a pass of its own would be another dispatch and another
// buffer.
//
//go:embed shaders/prepare_golem_q8.spv
var prepareGolemQ8SPIRV []byte

type prepareGolemPush struct {
	n       uint32
	columns uint32
}

// PrepareGolem is one site's activation transform, bound to the buffer it reads
// and writes.
type PrepareGolem struct {
	d     *Device
	n     int
	group int
	pre   *Buffer
	pipe  *Pipeline
	set   *Set
}

// NewPrepareGolem binds a site's vector to an activation buffer, transformed in
// place. pre is what the activation is multiplied by, one entry a column, and
// group is the width of the rotation that follows — the shader is built for
// 128, which is what the converter writes.
func NewPrepareGolem(d *Device, act *Buffer, pre []float32, group int) (*PrepareGolem, error) {
	return newPrepareGolem(d, nil, pre, group, prepareGolemSPIRV, []*Buffer{act, nil})
}

// NewPrepareGolemFromQ8 reads a Q8_0 activation — values and scales, the form
// the attention's mix arrives in — and writes the transformed floats into out.
func NewPrepareGolemFromQ8(d *Device, out, values, scales *Buffer, pre []float32, group int) (*PrepareGolem, error) {
	return newPrepareGolem(d, nil, pre, group, prepareGolemQ8SPIRV, []*Buffer{out, nil, values, scales})
}

// GolemPrepares is the transform's two pipelines, held once for a whole model.
//
// A site is a vector and a descriptor, not a pipeline: Qwen3.8 has four sites
// in each of sixty-five blocks, and building a pipeline for each would be two
// hundred and sixty compilations of two shaders. The vectors differ, and a
// vector is a buffer in a descriptor set.
type GolemPrepares struct {
	d      *Device
	plain  *Pipeline
	fromQ8 *Pipeline
}

// NewGolemPrepares compiles the two forms once.
func NewGolemPrepares(d *Device) (*GolemPrepares, error) {
	k := &GolemPrepares{d: d}
	var err error
	if k.plain, err = d.NewPipeline(prepareGolemSPIRV, 2, uint32(unsafe.Sizeof(prepareGolemPush{}))); err != nil {
		return nil, err
	}
	if k.fromQ8, err = d.NewPipeline(prepareGolemQ8SPIRV, 4, uint32(unsafe.Sizeof(prepareGolemPush{}))); err != nil {
		k.Close()
		return nil, err
	}
	return k, nil
}

// Bind is NewPrepareGolem against the shared pipeline: one site's vector over a
// buffer transformed in place.
func (k *GolemPrepares) Bind(act *Buffer, pre []float32, group int) (*PrepareGolem, error) {
	return newPrepareGolem(k.d, k.plain, pre, group, nil, []*Buffer{act, nil})
}

// BindFromQ8 is NewPrepareGolemFromQ8 against the shared pipeline.
func (k *GolemPrepares) BindFromQ8(out, values, scales *Buffer, pre []float32, group int) (*PrepareGolem, error) {
	return newPrepareGolem(k.d, k.fromQ8, pre, group, nil, []*Buffer{out, nil, values, scales})
}

func (k *GolemPrepares) Close() {
	for _, p := range []*Pipeline{k.plain, k.fromQ8} {
		if p != nil {
			p.Close()
		}
	}
	k.plain, k.fromQ8 = nil, nil
}

// newPrepareGolem binds a vector, either to a pipeline of its own built from
// spirv or to one the caller keeps. A PrepareGolem that did not build its
// pipeline does not close it.
func newPrepareGolem(d *Device, shared *Pipeline, pre []float32, group int, spirv []byte, bufs []*Buffer) (*PrepareGolem, error) {
	if group != prepareGolemGroup {
		return nil, fmt.Errorf("vk: the prepare kernel is built for a rotation of %d, asked for %d", prepareGolemGroup, group)
	}
	if len(pre)%group != 0 {
		return nil, fmt.Errorf("vk: a vector of %d does not divide into groups of %d", len(pre), group)
	}
	p := &PrepareGolem{d: d, n: len(pre), group: group}
	var err error
	if p.pre, err = d.Upload(asBytes(pre)); err != nil {
		return nil, err
	}
	buffers := append([]*Buffer(nil), bufs...)
	buffers[1] = p.pre // the vector is always the second binding
	pipe := shared
	if pipe == nil {
		if pipe, err = d.NewPipeline(spirv, len(buffers), uint32(unsafe.Sizeof(prepareGolemPush{}))); err != nil {
			p.Close()
			return nil, err
		}
		p.pipe = pipe
	}
	if p.set, err = pipe.NewSet(buffers); err != nil {
		p.Close()
		return nil, err
	}
	return p, nil
}

// prepareGolemGroup is the rotation width the shader was built for.
const prepareGolemGroup = 128

// Run transforms the first `columns` activations of the bound buffer in place.
func (p *PrepareGolem) Run(columns int) error {
	push := prepareGolemPush{n: uint32(p.n), columns: uint32(columns)}
	return p.set.Dispatch(uint32(columns*p.n/p.group), unsafe.Pointer(&push))
}

// Set is the descriptor a recording dispatches, for a caller that batches its
// own submissions.
func (p *PrepareGolem) Set() *Set { return p.set }

// Groups is how many workgroups Run dispatches for that many columns.
func (p *PrepareGolem) Groups(columns int) uint32 { return uint32(columns * p.n / p.group) }

// Push is the block Run would have sent.
func (p *PrepareGolem) Push(columns int) prepareGolemPush {
	return prepareGolemPush{n: uint32(p.n), columns: uint32(columns)}
}

func (p *PrepareGolem) Close() {
	if p.set != nil {
		p.set.Close()
	}
	if p.pipe != nil {
		p.pipe.Close()
	}
	if p.pre != nil {
		p.pre.Close()
	}
}
