package vk

// The golem product on the card: nn/pair.go's pair trellis.
//
// The weights go up once, as the file holds them. The step grid and the
// codebook go up once too, for the whole device and a tier — 256 floats, then
// the codebook's pairs — one copy serving every matrix of every model on the
// card. A state costs a window cut, a multiply, a bitfield extract and a
// shared-memory read, and it makes two weights.
//
// The activation arrives already through PrepareGolem: the per-column vector
// and the rotation belong to the site, not to the block, and both are undone on
// this side before the product rather than stored with the weights.

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O -DCOLUMNS=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_pair.comp -o shaders/matvec_h3g_1.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_pair.comp -o shaders/matvec_h3g_2.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_pair.comp -o shaders/matvec_h3g_4.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_pair.comp -o shaders/matvec_h3g_8.spv

//go:embed shaders/matvec_h3g_1.spv
var matvecH3G1SPIRV []byte

//go:embed shaders/matvec_h3g_2.spv
var matvecH3G2SPIRV []byte

//go:embed shaders/matvec_h3g_4.spv
var matvecH3G4SPIRV []byte

//go:embed shaders/matvec_h3g_8.spv
var matvecH3G8SPIRV []byte

//go:generate glslc -O -DKBITS=4 -DCOLUMNS=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_pair.comp -o shaders/matvec_h4g_1.spv
//go:generate glslc -O -DKBITS=4 -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_pair.comp -o shaders/matvec_h4g_2.spv
//go:generate glslc -O -DKBITS=4 -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_pair.comp -o shaders/matvec_h4g_4.spv
//go:generate glslc -O -DKBITS=4 -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_pair.comp -o shaders/matvec_h4g_8.spv

//go:embed shaders/matvec_h4g_1.spv
var matvecH4G1SPIRV []byte

//go:embed shaders/matvec_h4g_2.spv
var matvecH4G2SPIRV []byte

//go:embed shaders/matvec_h4g_4.spv
var matvecH4G4SPIRV []byte

//go:embed shaders/matvec_h4g_8.spv
var matvecH4G8SPIRV []byte

// golemRowsPerGroup is matvec_pair.comp's, at both tiers and every width: 256
// threads, sixteen lanes to a row group and two rows a lane.
const golemRowsPerGroup = 32

type golemPush struct {
	dim uint32 // outputs
	ffn uint32 // inputs
	col uint32 // the first column this dispatch answers
}

// golemReadTail is how far past a tensor the kernel's last block reaches: it
// reads a block's stream as words and slides the window across the one after
// it, so the last block of the last row wants less than a word of stream plus
// the word it is paired with. Sixteen bytes covers every tier.
const golemReadTail = 16

// GolemWidths are the pass widths the *mat-vec* is built for, and the reason
// is vk/qwen_pipeline.go's: a pass of two costs 1.056 of a pass of one because
// the weights are read once either way, and eight is where a mat-vec stops.
//
// Above them the shape changes rather than the width: vk/matmul_golem.go's
// tiled product decodes a weight once for a whole tile of the answer instead
// of once per column, and GolemTiledWidths is where a prompt goes. Both are
// bound to the same four bindings and the same push block, so a matrix holds
// one descriptor set a width whichever kernel reads it.
var GolemWidths = []int{1, 2, 4, 8}

// GolemKernels is everything about the golem product that belongs to the device
// rather than to a matrix: the step grid, and one pipeline a pass width.
type GolemKernels struct {
	d     *Device
	q     nn.Quant
	table *Buffer
	pipes map[int]*Pipeline
}

// NewGolemKernels builds the pipelines of one of golem's own formats, and
// whatever else belongs to the device rather than to a matrix: the step grid,
// uploaded once for the device and shared by every matrix of every model.
func NewGolemKernels(d *Device, q nn.Quant) (*GolemKernels, error) {
	spirv, ok := golemSPIRV(q)
	if !ok {
		return nil, fmt.Errorf("vk: there is no kernel for %s", q)
	}
	k := &GolemKernels{d: d, q: q, pipes: map[int]*Pipeline{}}
	var err error
	if k.table, err = d.Upload(golemTableFor(q)); err != nil {
		return nil, err
	}
	for _, columns := range GolemWidths {
		p, err := d.NewPipeline(spirv[columns], 4, golemPushSize)
		if err != nil {
			k.Close()
			return nil, err
		}
		k.pipes[columns] = p
	}
	// And the tiled product above them, which reads the same four bindings and
	// takes the same push block, so a matrix binds one set a width either way.
	tiled, ok := golemTiledSPIRV(q)
	if !ok {
		k.Close()
		return nil, fmt.Errorf("vk: there is no tiled kernel for %s", q)
	}
	for _, columns := range GolemTiledWidths {
		p, err := d.NewPipeline(tiled[columns], 4, golemPushSize)
		if err != nil {
			k.Close()
			return nil, err
		}
		k.pipes[columns] = p
	}
	return k, nil
}

func golemSPIRV(q nn.Quant) (map[int][]byte, bool) {
	switch q {
	case nn.H3G:
		return map[int][]byte{1: matvecH3G1SPIRV, 2: matvecH3G2SPIRV, 4: matvecH3G4SPIRV, 8: matvecH3G8SPIRV}, true
	case nn.H4G:
		return map[int][]byte{1: matvecH4G1SPIRV, 2: matvecH4G2SPIRV, 4: matvecH4G4SPIRV, 8: matvecH4G8SPIRV}, true
	}
	return nil, false
}

// golemTableFor is golemTable, and for a pair tier its codebook after it, as
// floats, which the kernels narrow back to the half2 each one is exactly.
func golemTableFor(q nn.Quant) []byte {
	out := golemTable()
	if p := nn.PairTierOf(q); p != nil {
		for _, v := range p.Codebook() {
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
		}
	}
	return out
}

// RowsPerGroup is how many rows a workgroup of a pass that wide answers.
func (k *GolemKernels) RowsPerGroup(columns int) int { return golemRowsPerGroup }

// golemTable is the head of what the second binding holds: the step grid, 256
// floats. golemTableFor puts the tier's codebook after it.
func golemTable() []byte {
	out := make([]byte, 256*4)
	for c := 0; c < 256; c++ {
		binary.LittleEndian.PutUint32(out[c*4:], math.Float32bits(nn.GolemStep(byte(c))))
	}
	return out
}

// Quant is the format these kernels read.
func (k *GolemKernels) Quant() nn.Quant { return k.q }

// Table is the step-grid buffer, for a caller binding its own sets.
func (k *GolemKernels) Table() *Buffer { return k.table }

// Pipeline is the kernel for a pass of that many columns.
func (k *GolemKernels) Pipeline(columns int) (*Pipeline, bool) {
	p, ok := k.pipes[columns]
	return p, ok
}

func (k *GolemKernels) Close() {
	for _, p := range k.pipes {
		p.Close()
	}
	k.pipes = nil
	if k.table != nil {
		k.table.Close()
		k.table = nil
	}
}

// GolemMatrix is one Golem weight matrix resident on the device, bound to the
// activation it reads and the output it writes.
type GolemMatrix struct {
	d          *Device
	k          *GolemKernels
	rows, cols int
	owned      []*Buffer // what this matrix allocated and must free

	weights *Buffer
	sets    map[int]*Set
	groups  map[int]uint32
}

// NewGolemMatrixOn uploads a matrix and binds it to buffers the caller owns:
// the activation it reads, already prepared, and the output it writes. This is
// the form a pipeline uses, where both are stages of a recording and neither is
// visible to the host.
func NewGolemMatrixOn(k *GolemKernels, data []byte, rows, cols int, act, out *Buffer) (*GolemMatrix, error) {
	// A path is the unit, not a block: a row that held half of one would have a
	// step with no codes under it.
	unit := nn.GolemSeq
	if cols%unit != 0 {
		return nil, fmt.Errorf("vk: a %s row needs a multiple of %d columns, given %d", k.q, unit, cols)
	}
	if want := rows * (nn.Matrix{Quant: k.q, Cols: cols}).RowBytes(); len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns need %d bytes, given %d", rows, cols, want, len(data))
	}
	m := &GolemMatrix{d: k.d, k: k, rows: rows, cols: cols, sets: map[int]*Set{}}
	var err error
	// The kernel reads a block's stream as words and slides the window across
	// the one after it, so the last block of the last row reaches past the
	// tensor by less than a word of the stream plus the word it is paired
	// with. Sixteen bytes covers every tier.
	if m.weights, err = k.d.UploadTail(data, golemReadTail); err != nil {
		return nil, err
	}
	m.owned = append(m.owned, m.weights)
	for columns, pipe := range k.pipes {
		if m.sets[columns], err = pipe.NewSet([]*Buffer{m.weights, k.table, act, out}); err != nil {
			m.Close()
			return nil, err
		}
	}
	m.groups = map[int]uint32{}
	for _, columns := range GolemWidths {
		per := k.RowsPerGroup(columns)
		m.groups[columns] = uint32((rows + per - 1) / per)
	}
	for _, columns := range GolemTiledWidths {
		m.groups[columns] = golemTiledGroups(rows, columns)
	}
	return m, nil
}

// newGolemMatrixShared binds weights another matrix already holds to a new pair
// of stages, so that the same tensor can be read from two places without being
// on the card twice. It owns nothing but its sets.
func newGolemMatrixShared(k *GolemKernels, weights *Buffer, rows, cols int, act, out *Buffer) (*GolemMatrix, error) {
	m := &GolemMatrix{d: k.d, k: k, rows: rows, cols: cols, sets: map[int]*Set{}, weights: weights}
	var err error
	for columns, pipe := range k.pipes {
		if m.sets[columns], err = pipe.NewSet([]*Buffer{weights, k.table, act, out}); err != nil {
			m.Close()
			return nil, err
		}
	}
	m.groups = map[int]uint32{}
	for _, columns := range GolemWidths {
		per := k.RowsPerGroup(columns)
		m.groups[columns] = uint32((rows + per - 1) / per)
	}
	for _, columns := range GolemTiledWidths {
		m.groups[columns] = golemTiledGroups(rows, columns)
	}
	return m, nil
}

// Set is the descriptor for a pass of that many columns, and Push the block
// that goes with it. A recording dispatches the two; MatVec below is what a
// caller with a host buffer does instead.
func (m *GolemMatrix) Set(columns int) *Set { return m.sets[columns] }

// Push is the block a dispatch sends, answering columns starting at first.
func (m *GolemMatrix) Push(first int) golemPush {
	return golemPush{dim: uint32(m.rows), ffn: uint32(m.cols), col: uint32(first)}
}

// Groups is how many workgroups a pass of that width needs. It is not one
// number for the matrix: the widths are built for different workgroups.
func (m *GolemMatrix) Groups(columns int) uint32 { return m.groups[columns] }

func (m *GolemMatrix) Close() {
	for _, s := range m.sets {
		s.Close()
	}
	m.sets = nil
	for _, b := range m.owned {
		b.Close()
	}
	m.owned = nil
}

// hostGolemMatrix is a matrix with its own host-visible activation and output,
// for a caller outside a pipeline: a test that wants one product, or a
// benchmark that wants to time one. Everything in a model goes through
// NewGolemMatrixOn instead, where both ends are stages of a recording and the
// host never sees them.
type hostGolemMatrix struct {
	*GolemMatrix
	k   *GolemKernels
	act *Buffer
	out *Buffer
}

// newHostGolemMatrix is the same for whichever of golem's formats wrote the
// bytes.
func newHostGolemMatrix(d *Device, data []byte, rows, cols int, q nn.Quant) (*hostGolemMatrix, error) {
	k, err := NewGolemKernels(d, q)
	if err != nil {
		return nil, err
	}
	h := &hostGolemMatrix{k: k}
	if h.act, err = d.Host(cols*4, bufferUsageStorage); err != nil {
		h.Close()
		return nil, err
	}
	if h.out, err = d.Readback(rows*4, bufferUsageStorage); err != nil {
		h.Close()
		return nil, err
	}
	if h.GolemMatrix, err = NewGolemMatrixOn(k, data, rows, cols, h.act, h.out); err != nil {
		h.Close()
		return nil, err
	}
	return h, nil
}

// MatVec computes y = W*x for one activation, which must already have been
// through PrepareGolem.
func (h *hostGolemMatrix) MatVec(x []float32, out []float32) error {
	if len(x) != h.cols {
		return fmt.Errorf("vk: the matrix reads %d inputs, given %d", h.cols, len(x))
	}
	if len(out) != h.rows {
		return fmt.Errorf("vk: the matrix writes %d outputs, given %d", h.rows, len(out))
	}
	copy(h.act.Floats(), x)
	push := h.Push(0)
	if err := h.Set(1).Dispatch(h.Groups(1), unsafe.Pointer(&push)); err != nil {
		return err
	}
	copy(out, h.out.Floats())
	return nil
}

func (h *hostGolemMatrix) Close() {
	if h.GolemMatrix != nil {
		h.GolemMatrix.Close()
		h.GolemMatrix = nil
	}
	for _, b := range []*Buffer{h.act, h.out} {
		if b != nil {
			b.Close()
		}
	}
	h.act, h.out = nil, nil
	if h.k != nil {
		h.k.Close()
		h.k = nil
	}
}
