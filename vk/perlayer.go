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
// at the head of the pass. After PerLayerInputs the caller writes only the
// per-layer embedding's rows there, and the card adds the projection of the
// stream to them: a bfloat16 product of 8960 rows that cost the processor 110
// ms for a pass of 512 columns.

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/ple_act.comp -o shaders/ple_act.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/ple_inputs.comp -o shaders/ple_inputs.spv
//go:generate glslc -O -DROUND --target-env=vulkan1.1 -fshader-stage=compute shaders/ple_inputs.comp -o shaders/ple_round.spv
//go:embed shaders/ple_act.spv
var pleActSPIRV []byte

//go:embed shaders/ple_inputs.spv
var pleInputsSPIRV []byte

//go:embed shaders/ple_round.spv
var pleRoundSPIRV []byte

// pleInputsPush is shaders/ple_inputs.comp's.
type pleInputsPush struct {
	width, row                uint32
	eps, lookup, project, sum float32
}

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

	// The projection half of the inputs, when the card makes it: the stream
	// rounded to bfloat16, the product, and the sum with the lookups.
	dim                          int
	push                         pleInputsPush
	round, product, combine      *Pipeline
	weights, gain, rounded, pj   *Buffer
	roundSet, productSet, sumSet *Set
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

// perLayerInputs has the card make the projection half of the inputs out of
// stream, which holds the pass's embeddings when the head of the pass reads
// it: proj is per_layer_model_proj in bfloat16, a slice's width times the
// blocks in rows and the stream's width in columns, and gain the norm every
// slice of it goes through. The stage then holds the lookups alone.
func (m *Mixture) perLayerInputs(stream *Buffer, proj []byte, gain []float32, eps float32) error {
	p := m.perLayer
	if p == nil {
		return fmt.Errorf("vk: the per-layer inputs before PerLayer")
	}
	rows := p.width * p.blocks
	if len(proj) != rows*m.dim*2 || len(gain) != p.width {
		return fmt.Errorf("vk: a projection of %d bytes and a gain of %d for %d rows of %d", len(proj), len(gain), rows, m.dim)
	}
	p.dim = m.dim
	p.push = pleInputsPush{width: uint32(p.width), row: uint32(rows), eps: eps,
		lookup: float32(math.Sqrt(float64(p.width))), project: float32(1 / math.Sqrt(float64(m.dim))),
		sum: float32(1 / math.Sqrt2)}
	var err error
	if p.round, err = m.d.NewPipeline(pleRoundSPIRV, 2, 4); err != nil {
		return err
	}
	if p.combine, err = m.d.NewPipeline(pleInputsSPIRV, 3, uint32(unsafe.Sizeof(pleInputsPush{}))); err != nil {
		return err
	}
	if p.product, err = m.d.NewPipeline(matvecBF16SPIRV, 3, uint32(unsafe.Sizeof(matvecKPush{}))); err != nil {
		return err
	}
	for _, w := range []struct {
		columns int
		spirv   []byte
	}{{2, matvecBF16_2SPIRV}, {4, matvecBF16_4SPIRV}, {8, matvecBF16_8SPIRV}, {16, matvecBF16_16SPIRV}} {
		if err := p.product.Wide(w.columns, w.spirv); err != nil {
			return err
		}
	}
	if p.weights, err = m.d.Upload(proj); err != nil {
		return err
	}
	if p.gain, err = m.d.Upload(asBytes(gain)); err != nil {
		return err
	}
	if p.rounded, err = m.d.Local(m.dim*4*maxColumns, bufferUsageStorage); err != nil {
		return err
	}
	if p.pj, err = m.d.Local(rows*4*maxColumns, bufferUsageStorage); err != nil {
		return err
	}
	if p.roundSet, err = p.round.NewSet([]*Buffer{stream, p.rounded}); err != nil {
		return err
	}
	if p.productSet, err = p.product.NewSet([]*Buffer{p.weights, p.rounded, p.pj}); err != nil {
		return err
	}
	p.sumSet, err = p.combine.NewSet([]*Buffer{p.pj, p.gain, p.inputs})
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
// block of it, and makes their projection half there when it is the card's
// to make. The stream has to hold the pass's embeddings by then.
func (m *Mixture) recordPerLayerInputs(r *Recorder, columns int) {
	p := m.perLayer
	rows := p.width * p.blocks
	r.CopyFrom(p.inputs, 0, p.stage, 0, columns*rows*4)
	if p.product == nil {
		return
	}
	n := uint32(columns * p.dim)
	r.Dispatch(p.roundSet, (n+255)/256, unsafe.Pointer(&n))
	r.Barrier()
	// The widest binary that fits, as many times as the pass needs: a product
	// of 8960 rows is half a megabyte of bfloat16 a column group, and the card
	// reads it from its cache after the first.
	groups := uint32((rows + 15) / 16)
	for c := 0; c < columns; {
		w := max(1, p.product.widestUpTo(columns-c))
		push := matvecKPush{Dim: uint32(rows), FFN: uint32(p.dim), Col: uint32(c)}
		if w == 1 {
			r.Dispatch(p.productSet, groups, unsafe.Pointer(&push))
		} else {
			r.DispatchWide(p.productSet, w, groups, unsafe.Pointer(&push))
		}
		c += w
	}
	r.Barrier()
	r.DispatchColumns(p.sumSet, uint32(p.blocks), uint32(columns), unsafe.Pointer(&p.push))
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
	for _, s := range []*Set{p.actSet, p.roundSet, p.productSet, p.sumSet} {
		if s != nil {
			s.Close()
		}
	}
	for _, x := range []*Buffer{p.stage, p.inputs, p.gate, p.weights, p.gain, p.rounded, p.pj} {
		if x != nil {
			x.Close()
		}
	}
	for _, x := range []*Pipeline{p.act, p.round, p.product, p.combine} {
		if x != nil {
			x.Close()
		}
	}
}
