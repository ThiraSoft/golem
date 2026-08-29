package vk

// The activation side of the D4G scheme on the card.
//
// A D4G matrix is stored as A·(q ⊙ W): its columns scaled by the site's vector
// and then rotated. The product is only the right one if the activation meets
// the reciprocal of that scale and the same rotation, and neither belongs to
// the weights — one vector serves every matrix of a site, and the rotation is
// the same for all of them. So it is done once, here, to the activation, and
// the weights carry nothing about it.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/prepare_d4g.comp -o shaders/prepare_d4g.spv

//go:embed shaders/prepare_d4g.spv
var prepareD4GSPIRV []byte

type prepareD4GPush struct {
	n       uint32
	columns uint32
}

// PrepareD4G is one site's activation transform, bound to the buffer it reads
// and writes.
type PrepareD4G struct {
	d     *Device
	n     int
	group int
	pre   *Buffer
	pipe  *Pipeline
	set   *Set
}

// NewPrepareD4G binds a site's vector to an activation buffer. pre is what the
// activation is multiplied by, one entry a column, and group is the width of
// the rotation that follows — the shader is built for 128, which is what the
// converter writes.
func NewPrepareD4G(d *Device, act *Buffer, pre []float32, group int) (*PrepareD4G, error) {
	if group != prepareD4GGroup {
		return nil, fmt.Errorf("vk: the prepare kernel is built for a rotation of %d, asked for %d", prepareD4GGroup, group)
	}
	if len(pre)%group != 0 {
		return nil, fmt.Errorf("vk: a vector of %d does not divide into groups of %d", len(pre), group)
	}
	p := &PrepareD4G{d: d, n: len(pre), group: group}
	var err error
	if p.pre, err = d.Upload(asBytes(pre)); err != nil {
		return nil, err
	}
	buffers := []*Buffer{act, p.pre}
	if p.pipe, err = d.NewPipeline(prepareD4GSPIRV, len(buffers), uint32(unsafe.Sizeof(prepareD4GPush{}))); err != nil {
		p.Close()
		return nil, err
	}
	if p.set, err = p.pipe.NewSet(buffers); err != nil {
		p.Close()
		return nil, err
	}
	return p, nil
}

// prepareD4GGroup is the rotation width the shader was built for.
const prepareD4GGroup = 128

// Run transforms the first `columns` activations of the bound buffer in place.
func (p *PrepareD4G) Run(columns int) error {
	push := prepareD4GPush{n: uint32(p.n), columns: uint32(columns)}
	return p.set.Dispatch(uint32(columns*p.n/p.group), unsafe.Pointer(&push))
}

// Set is the descriptor a recording dispatches, for a caller that batches its
// own submissions.
func (p *PrepareD4G) Set() *Set { return p.set }

// Groups is how many workgroups Run dispatches for that many columns.
func (p *PrepareD4G) Groups(columns int) uint32 { return uint32(columns * p.n / p.group) }

// Push is the block Run would have sent.
func (p *PrepareD4G) Push(columns int) prepareD4GPush {
	return prepareD4GPush{n: uint32(p.n), columns: uint32(columns)}
}

func (p *PrepareD4G) Close() {
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
