package vk

// The golem product on the card: a trellis, decoded without a codebook.
//
// The weights go up once, as the file holds them. The step grid goes up once
// too, for the whole device — 256 floats, one copy serving every matrix of
// every model on the card. A code costs a window read, a multiply and a byte
// sum, and what it saves is a third of the bytes a Q4_K row would have cost to
// read.
//
// The activation arrives already through PrepareD4G: the per-column vector and
// the rotation belong to the site, not to the block, and both are undone on
// this side before the product rather than stored with the weights.

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t4g.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t4g_2.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t4g_4.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t4g_8.spv

//go:embed shaders/matvec_t4g.spv
var matvecT4GSPIRV []byte

//go:embed shaders/matvec_t4g_2.spv
var matvecT4G2SPIRV []byte

//go:embed shaders/matvec_t4g_4.spv
var matvecT4G4SPIRV []byte

//go:embed shaders/matvec_t4g_8.spv
var matvecT4G8SPIRV []byte

//go:generate glslc -O -DKBITS=3 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t3g.spv
//go:generate glslc -O -DKBITS=3 -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t3g_2.spv
//go:generate glslc -O -DKBITS=3 -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t3g_4.spv
//go:generate glslc -O -DKBITS=3 -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t3g_8.spv

//go:embed shaders/matvec_t3g.spv
var matvecT3GSPIRV []byte

//go:embed shaders/matvec_t3g_2.spv
var matvecT3G2SPIRV []byte

//go:embed shaders/matvec_t3g_4.spv
var matvecT3G4SPIRV []byte

//go:embed shaders/matvec_t3g_8.spv
var matvecT3G8SPIRV []byte

//go:generate glslc -O -DKBITS=5 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t5g.spv
//go:generate glslc -O -DKBITS=5 -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t5g_2.spv
//go:generate glslc -O -DKBITS=5 -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t5g_4.spv
//go:generate glslc -O -DKBITS=5 -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_t4g.comp -o shaders/matvec_t5g_8.spv

//go:embed shaders/matvec_t5g.spv
var matvecT5GSPIRV []byte

//go:embed shaders/matvec_t5g_2.spv
var matvecT5G2SPIRV []byte

//go:embed shaders/matvec_t5g_4.spv
var matvecT5G4SPIRV []byte

//go:embed shaders/matvec_t5g_8.spv
var matvecT5G8SPIRV []byte

type d4gPush struct {
	dim uint32 // outputs
	ffn uint32 // inputs
	col uint32 // the first column this dispatch answers
}

// d4gRowsPerGroup is the shader's OUTS: a workgroup of 128 answers sixteen
// rows, eight lanes to a row.
const d4gRowsPerGroup = 16

// D4GWidths are the pass widths the kernels are built for, and the reason is
// vk/qwen_pipeline.go's: a pass of two costs 1.056 of a pass of one because the
// weights are read once either way, and eight is where a mat-vec stops.
var D4GWidths = []int{1, 2, 4, 8}

// D4GKernels is everything about the golem product that belongs to the device
// rather than to a matrix: the step grid, and one pipeline a pass width.
type D4GKernels struct {
	d     *Device
	q     nn.Quant
	table *Buffer
	pipes map[int]*Pipeline
}

// NewGolemKernels builds the pipelines of one of golem's own formats, and
// whatever else belongs to the device rather than to a matrix: the step grid,
// uploaded once for the device and shared by every matrix of every model.
func NewGolemKernels(d *Device, q nn.Quant) (*D4GKernels, error) {
	spirv, ok := golemSPIRV(q)
	if !ok {
		return nil, fmt.Errorf("vk: there is no kernel for %s", q)
	}
	k := &D4GKernels{d: d, q: q, pipes: map[int]*Pipeline{}}
	var err error
	if k.table, err = d.Upload(golemTable()); err != nil {
		return nil, err
	}
	for _, columns := range D4GWidths {
		p, err := d.NewPipeline(spirv[columns], 4, uint32(unsafe.Sizeof(d4gPush{})))
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
	case nn.T3G:
		return map[int][]byte{1: matvecT3GSPIRV, 2: matvecT3G2SPIRV, 4: matvecT3G4SPIRV, 8: matvecT3G8SPIRV}, true
	case nn.T4G:
		return map[int][]byte{1: matvecT4GSPIRV, 2: matvecT4G2SPIRV, 4: matvecT4G4SPIRV, 8: matvecT4G8SPIRV}, true
	case nn.T5G:
		return map[int][]byte{1: matvecT5GSPIRV, 2: matvecT5G2SPIRV, 4: matvecT5G4SPIRV, 8: matvecT5G8SPIRV}, true
	}
	return nil, false
}

// golemTable is what the second binding holds: the step grid, 256 floats, the
// only table a trellis has. Uploaded once for the device and shared by every
// matrix of every model.
func golemTable() []byte {
	out := make([]byte, 256*4)
	for c := 0; c < 256; c++ {
		binary.LittleEndian.PutUint32(out[c*4:], math.Float32bits(nn.T4GStep(byte(c))))
	}
	return out
}

// Quant is the format these kernels read.
func (k *D4GKernels) Quant() nn.Quant { return k.q }

// Table is the step-grid buffer, for a caller binding its own sets.
func (k *D4GKernels) Table() *Buffer { return k.table }

// Pipeline is the kernel for a pass of that many columns.
func (k *D4GKernels) Pipeline(columns int) (*Pipeline, bool) {
	p, ok := k.pipes[columns]
	return p, ok
}

func (k *D4GKernels) Close() {
	for _, p := range k.pipes {
		p.Close()
	}
	k.pipes = nil
	if k.table != nil {
		k.table.Close()
		k.table = nil
	}
}

// D4GMatrix is one D4G weight matrix resident on the device, bound to the
// activation it reads and the output it writes.
type D4GMatrix struct {
	d          *Device
	k          *D4GKernels
	rows, cols int
	owned      []*Buffer // what this matrix allocated and must free

	weights *Buffer
	sets    map[int]*Set
	groups  uint32
}

// NewD4GMatrixOn uploads a matrix and binds it to buffers the caller owns: the
// activation it reads, already prepared, and the output it writes. This is the
// form a pipeline uses, where both are stages of a recording and neither is
// visible to the host.
func NewD4GMatrixOn(k *D4GKernels, data []byte, rows, cols int, act, out *Buffer) (*D4GMatrix, error) {
	// A path is the unit, not a block: a row that held half of one would have a
	// step with no codes under it.
	unit := nn.T4GSeq
	if cols%unit != 0 {
		return nil, fmt.Errorf("vk: a %s row needs a multiple of %d columns, given %d", k.q, unit, cols)
	}
	if want := rows * (nn.Matrix{Quant: k.q, Cols: cols}).RowBytes(); len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns need %d bytes, given %d", rows, cols, want, len(data))
	}
	m := &D4GMatrix{d: k.d, k: k, rows: rows, cols: cols, sets: map[int]*Set{}}
	var err error
	if m.weights, err = k.d.Upload(data); err != nil {
		return nil, err
	}
	m.owned = append(m.owned, m.weights)
	for _, columns := range D4GWidths {
		pipe := k.pipes[columns]
		if m.sets[columns], err = pipe.NewSet([]*Buffer{m.weights, k.table, act, out}); err != nil {
			m.Close()
			return nil, err
		}
	}
	m.groups = uint32((rows + d4gRowsPerGroup - 1) / d4gRowsPerGroup)
	return m, nil
}

// Set is the descriptor for a pass of that many columns, and Push the block
// that goes with it. A recording dispatches the two; MatVec below is what a
// caller with a host buffer does instead.
func (m *D4GMatrix) Set(columns int) *Set { return m.sets[columns] }

// Push is the block a dispatch sends, answering columns starting at first.
func (m *D4GMatrix) Push(first int) d4gPush {
	return d4gPush{dim: uint32(m.rows), ffn: uint32(m.cols), col: uint32(first)}
}

// Groups is how many workgroups one dispatch needs.
func (m *D4GMatrix) Groups() uint32 { return m.groups }

func (m *D4GMatrix) Close() {
	for _, s := range m.sets {
		s.Close()
	}
	m.sets = nil
	for _, b := range m.owned {
		b.Close()
	}
	m.owned = nil
}

// hostD4GMatrix is a matrix with its own host-visible activation and output,
// for a caller outside a pipeline: a test that wants one product, or a
// benchmark that wants to time one. Everything in a model goes through
// NewD4GMatrixOn instead, where both ends are stages of a recording and the
// host never sees them.
type hostD4GMatrix struct {
	*D4GMatrix
	k   *D4GKernels
	act *Buffer
	out *Buffer
}

// newHostGolemMatrix is the same for whichever of golem's formats wrote the
// bytes.
func newHostGolemMatrix(d *Device, data []byte, rows, cols int, q nn.Quant) (*hostD4GMatrix, error) {
	k, err := NewGolemKernels(d, q)
	if err != nil {
		return nil, err
	}
	h := &hostD4GMatrix{k: k}
	if h.act, err = d.Host(cols*4, bufferUsageStorage); err != nil {
		h.Close()
		return nil, err
	}
	if h.out, err = d.Readback(rows*4, bufferUsageStorage); err != nil {
		h.Close()
		return nil, err
	}
	if h.D4GMatrix, err = NewD4GMatrixOn(k, data, rows, cols, h.act, h.out); err != nil {
		h.Close()
		return nil, err
	}
	return h, nil
}

// MatVec computes y = W*x for one activation, which must already have been
// through PrepareD4G.
func (h *hostD4GMatrix) MatVec(x []float32, out []float32) error {
	if len(x) != h.cols {
		return fmt.Errorf("vk: the matrix reads %d inputs, given %d", h.cols, len(x))
	}
	if len(out) != h.rows {
		return fmt.Errorf("vk: the matrix writes %d outputs, given %d", h.rows, len(out))
	}
	copy(h.act.Floats(), x)
	push := h.Push(0)
	if err := h.Set(1).Dispatch(h.Groups(), unsafe.Pointer(&push)); err != nil {
		return err
	}
	copy(out, h.out.Floats())
	return nil
}

func (h *hostD4GMatrix) Close() {
	if h.D4GMatrix != nil {
		h.D4GMatrix.Close()
		h.D4GMatrix = nil
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
