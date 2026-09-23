package vk

// Activation rotation and Q8_0 quantization on the card for Prism Bonsai.
//
// In Bonsai models, activations undergo a sign flip and normalized Walsh-Hadamard
// transform before multiplying against ternary weights. This kernel performs that
// transform and quantizes the result to Q8_0 in a single compute pass.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/rotate_q8.comp -o shaders/rotate_q8.spv
//go:embed shaders/rotate_q8.spv
var rotateQ8SPIRV []byte

//go:generate glslc -O -DGATHER --target-env=vulkan1.1 -fshader-stage=compute shaders/rotate_q8.comp -o shaders/rotate_q8_gather.spv
//go:embed shaders/rotate_q8_gather.spv
var rotateQ8GatherSPIRV []byte

// RotateQ8Group is the group width of the Walsh-Hadamard transform in rotate_q8.comp.
const RotateQ8Group = 1024

type rotateQ8Push struct {
	n       uint32
	columns uint32
}

// RotateQ8Groups returns the number of workgroups needed for n values and columns.
func RotateQ8Groups(n, columns int) uint32 {
	return uint32(columns * (n / RotateQ8Group))
}

// RotateQ8Pipelines holds the plain and gathered rotation pipelines for a device.
type RotateQ8Pipelines struct {
	d      *Device
	plain  *Pipeline
	gather *Pipeline
}

// NewRotateQ8Pipelines compiles the plain and gather rotation pipelines.
func NewRotateQ8Pipelines(d *Device) (*RotateQ8Pipelines, error) {
	k := &RotateQ8Pipelines{d: d}
	var err error
	if k.plain, err = d.NewPipeline(rotateQ8SPIRV, 4, uint32(unsafe.Sizeof(rotateQ8Push{}))); err != nil {
		return nil, err
	}
	if k.gather, err = d.NewPipeline(rotateQ8GatherSPIRV, 5, uint32(unsafe.Sizeof(rotateQ8Push{}))); err != nil {
		k.Close()
		return nil, err
	}
	return k, nil
}

// Plain returns the plain rotation pipeline without index gathering.
func (k *RotateQ8Pipelines) Plain() *Pipeline { return k.plain }

// Gather returns the rotation pipeline with index gathering enabled.
func (k *RotateQ8Pipelines) Gather() *Pipeline { return k.gather }

// Bind creates a descriptor set for the plain rotation pipeline.
func (k *RotateQ8Pipelines) Bind(src, signs, aq, as *Buffer) (*Set, error) {
	return k.plain.NewSet([]*Buffer{src, signs, aq, as})
}

// BindGather creates a descriptor set for the gather rotation pipeline.
func (k *RotateQ8Pipelines) BindGather(src, signs, aq, as, gather *Buffer) (*Set, error) {
	return k.gather.NewSet([]*Buffer{src, signs, aq, as, gather})
}

// Close releases both pipelines.
func (k *RotateQ8Pipelines) Close() {
	if k.plain != nil {
		k.plain.Close()
		k.plain = nil
	}
	if k.gather != nil {
		k.gather.Close()
		k.gather = nil
	}
}

// RotateQ8 manages a rotation pass bound to buffers.
type RotateQ8 struct {
	d       *Device
	n       int
	pipe    *Pipeline
	set     *Set
	signs   *Buffer
	gather  *Buffer
	ownPipe bool
}

// NewRotateQ8 binds a plain rotation pass over an activation buffer.
func NewRotateQ8(d *Device, src, aq, as *Buffer, signs []float32) (*RotateQ8, error) {
	if len(signs)%RotateQ8Group != 0 {
		return nil, fmt.Errorf("vk: rotate vector length %d must be a multiple of %d", len(signs), RotateQ8Group)
	}
	r := &RotateQ8{d: d, n: len(signs)}
	var err error
	if r.signs, err = d.Upload(asBytes(signs)); err != nil {
		return nil, err
	}
	if r.pipe, err = d.NewPipeline(rotateQ8SPIRV, 4, uint32(unsafe.Sizeof(rotateQ8Push{}))); err != nil {
		r.Close()
		return nil, err
	}
	r.ownPipe = true
	if r.set, err = r.pipe.NewSet([]*Buffer{src, r.signs, aq, as}); err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

// NewRotateQ8Gather binds a gathered rotation pass over an activation buffer.
func NewRotateQ8Gather(d *Device, src, aq, as *Buffer, signs []float32, gather []int32) (*RotateQ8, error) {
	if len(signs)%RotateQ8Group != 0 {
		return nil, fmt.Errorf("vk: rotate vector length %d must be a multiple of %d", len(signs), RotateQ8Group)
	}
	if len(gather) != len(signs) {
		return nil, fmt.Errorf("vk: gather length %d must match signs length %d", len(gather), len(signs))
	}
	r := &RotateQ8{d: d, n: len(signs)}
	var err error
	if r.signs, err = d.Upload(asBytes(signs)); err != nil {
		return nil, err
	}
	if r.gather, err = d.Upload(asBytesInt32(gather)); err != nil {
		r.Close()
		return nil, err
	}
	if r.pipe, err = d.NewPipeline(rotateQ8GatherSPIRV, 5, uint32(unsafe.Sizeof(rotateQ8Push{}))); err != nil {
		r.Close()
		return nil, err
	}
	r.ownPipe = true
	if r.set, err = r.pipe.NewSet([]*Buffer{src, r.signs, aq, as, r.gather}); err != nil {
		r.Close()
		return nil, err
	}
	return r, nil
}

// Run executes the rotation and quantization pass for columns.
func (r *RotateQ8) Run(columns int) error {
	push := rotateQ8Push{n: uint32(r.n), columns: uint32(columns)}
	groups := RotateQ8Groups(r.n, columns)
	return r.set.Dispatch(groups, unsafe.Pointer(&push))
}

// Set returns the descriptor set for custom command recording.
func (r *RotateQ8) Set() *Set { return r.set }

// Close frees buffers and pipelines created for this instance.
func (r *RotateQ8) Close() {
	if r.set != nil {
		r.set.Close()
		r.set = nil
	}
	if r.ownPipe && r.pipe != nil {
		r.pipe.Close()
		r.pipe = nil
	}
	if r.signs != nil {
		r.signs.Close()
		r.signs = nil
	}
	if r.gather != nil {
		r.gather.Close()
		r.gather = nil
	}
}
