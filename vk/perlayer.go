package vk

// Gemma 4's per-layer branch, the part of it that is products.
//
// E2B and E4B carry a second embedding beside the first: every token has a row
// of 256 values for each of its blocks, and every block folds its own slice
// back into the stream after the feed forward. gemma/block.go says it in Go:
// the stream unnormed into Q8_0, an input gate of 256 rows over it, the GELU
// of that times the block's slice, a projection of it back to the stream's
// width, and that projection normed and added on. The end, the norm and the
// residual, is shaders/combine.comp's dense layout under another gain, and
// vk/stack.go records it; what is here is the three kernels in between, on
// the mixture because the mixture already holds what they need: the product
// pipelines, ggml's GELU table, and buffers between two kernels that the
// shared branch is done with by then.
//
// The inputs themselves, a row of every block's slices for each column of a
// pass, are written by the caller into PerLayerStage and copied onto the card
// at the head of the pass.

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/ple_act.comp -o shaders/ple_act.spv
//go:embed shaders/ple_act.spv
var pleActSPIRV []byte

// pleActPush is shaders/ple_act.comp's.
type pleActPush struct {
	ffn, block, stride uint32
}

type perLayer struct {
	width, blocks int // a block's slice, and how many slices a row holds
	act           *Pipeline
	actSet        *Set
	stage         *Buffer // what the caller writes, a row a column
	inputs        *Buffer // the same on the card
	gate          *Buffer // the input gate's product, as floats
	added         []*perLayerBlock
}

type perLayerBlock struct {
	gate, proj     *Buffer
	setUp, setDown *Set
}

// PerLayer gives the mixture a per-layer branch of the given width in each of
// blocks blocks. It is called once, before AddPerLayer.
func (m *Mixture) PerLayer(width, blocks int) error {
	if m.perLayer != nil {
		return fmt.Errorf("vk: the mixture already has a per-layer branch")
	}
	if width%nn.QuantBlock != 0 || width > m.dense {
		return fmt.Errorf("vk: a per-layer branch %d wide beside a shared branch of %d", width, m.dense)
	}
	p := &perLayer{width: width, blocks: blocks}
	m.perLayer = p
	var err error
	if p.act, err = m.d.NewPipeline(pleActSPIRV, 5, uint32(unsafe.Sizeof(pleActPush{}))); err != nil {
		return err
	}
	row := width * blocks * 4
	if p.stage, err = m.d.Readback(row*maxColumns, bufferUsageTransferSrc|bufferUsageTransferDst); err != nil {
		return err
	}
	if p.inputs, err = m.d.Local(row*maxColumns, bufferUsageStorage|bufferUsageTransferDst); err != nil {
		return err
	}
	if p.gate, err = m.d.Local(width*4*maxColumns, bufferUsageStorage); err != nil {
		return err
	}
	// The product goes where the shared branch's intermediate went: it is
	// read by the projection exactly as that one is read by the down.
	p.actSet, err = p.act.NewSet([]*Buffer{p.gate, m.gelu, p.inputs, m.daq, m.das})
	return err
}

// AddPerLayer uploads one block's input gate (width rows over the stream) and
// projection (the stream's rows over width), in the order the blocks run.
func (m *Mixture) AddPerLayer(q nn.Quant, gate, proj []byte) error {
	p := m.perLayer
	if p == nil {
		return fmt.Errorf("vk: AddPerLayer before PerLayer")
	}
	if len(p.added) == p.blocks {
		return fmt.Errorf("vk: a per-layer branch for %d blocks, given another", p.blocks)
	}
	prod, err := m.products.get(q)
	if err != nil {
		return fmt.Errorf("vk: the per-layer branch: %w", err)
	}
	b := &perLayerBlock{}
	p.added = append(p.added, b)
	for _, spec := range []struct {
		what       string
		data       []byte
		rows, cols int
		into       **Buffer
	}{
		{"input gate", gate, p.width, m.dim, &b.gate},
		{"projection", proj, m.dim, p.width, &b.proj},
	} {
		layout, err := quantLayout(q, spec.data, spec.rows, spec.cols)
		if err != nil {
			return fmt.Errorf("vk: the per-layer %s: %w", spec.what, err)
		}
		if *spec.into, err = m.d.Upload(layout); err != nil {
			return err
		}
	}
	if b.setUp, err = prod.NewSet([]*Buffer{b.gate, m.dxq, m.dxs, p.gate}); err != nil {
		return err
	}
	b.setDown, err = prod.NewSet([]*Buffer{b.proj, m.daq, m.das, m.dout})
	return err
}

// PerLayerStage is where the caller writes a pass's per-layer inputs: for each
// column, every block's slice one after the other. Nil without a branch.
func (m *Mixture) PerLayerStage() []float32 {
	if m.perLayer == nil {
		return nil
	}
	return m.perLayer.stage.Floats()
}

// recordPerLayerInputs copies a pass's inputs onto the card, once for every
// block of it.
func (m *Mixture) recordPerLayerInputs(r *Recorder, columns int) {
	p := m.perLayer
	r.CopyFrom(p.inputs, 0, p.stage, 0, columns*p.width*p.blocks*4)
}

// recordPerLayer runs one block's branch from the stream's Q8_0 form, which
// the caller left in SharedInput, to the projection, which it leaves in
// Outputs' first buffer.
func (m *Mixture) recordPerLayer(r *Recorder, block, columns int) {
	p := m.perLayer
	b := p.added[block]
	width := passWidth(columns)
	product := func(set *Set, rows, cols int) {
		push := moePush{dim: uint32(rows), ffn: uint32(cols), used: 1, split: 1}
		groups := m.productGroups(width, rows)
		if width > 1 {
			r.DispatchWide(set, width, groups, unsafe.Pointer(&push))
		} else {
			r.Dispatch(set, groups, unsafe.Pointer(&push))
		}
	}
	product(b.setUp, p.width, m.dim)
	r.Barrier()
	act := pleActPush{ffn: uint32(p.width), block: uint32(block), stride: uint32(p.width * p.blocks)}
	r.DispatchColumns(p.actSet, uint32((p.width+255)/256), uint32(width), unsafe.Pointer(&act))
	r.Barrier()
	product(b.setDown, m.dim, p.width)
}

func (p *perLayer) close() {
	if p == nil {
		return
	}
	for _, b := range p.added {
		for _, s := range []**Set{&b.setUp, &b.setDown} {
			if *s != nil {
				(*s).Close()
				*s = nil
			}
		}
		for _, x := range []**Buffer{&b.gate, &b.proj} {
			if *x != nil {
				(*x).Close()
				*x = nil
			}
		}
	}
	p.added = nil
	if p.actSet != nil {
		p.actSet.Close()
	}
	for _, x := range []*Buffer{p.stage, p.inputs, p.gate} {
		if x != nil {
			x.Close()
		}
	}
	for _, x := range []*Pipeline{p.act} {
		if x != nil {
			x.Close()
		}
	}
}
