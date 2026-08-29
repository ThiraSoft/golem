package vk

// The D4G product on the card.
//
// The weights go up once, as the file holds them. The lattice goes up once too,
// for the whole device — the table is the lattice and not a codebook fitted to
// a tensor, so one copy serves every matrix of every model on the card, which
// is sixteen kibibytes at twelve bits a code and a quarter of a mebibyte at
// sixteen. That is the reason this format is worth a kernel at all: a code
// costs one lookup and four sign extensions, and what it saves is a third of
// the bytes a Q4_K row would have cost to read.
//
// The activation arrives already through PrepareD4G: the per-column vector and
// the rotation belong to the site, not to the block, and both are undone on
// this side before the product rather than stored with the weights.

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g_2.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g_4.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g_8.spv
//go:generate glslc -O -DBITS16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g16.spv
//go:generate glslc -O -DBITS16 -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g16_2.spv
//go:generate glslc -O -DBITS16 -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g16_4.spv
//go:generate glslc -O -DBITS16 -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g16_8.spv
//go:generate glslc -O -DNOTABLE --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g_notable.spv

//go:embed shaders/matvec_d4g.spv
var matvecD4GSPIRV []byte

//go:embed shaders/matvec_d4g_2.spv
var matvecD4G2SPIRV []byte

//go:embed shaders/matvec_d4g_4.spv
var matvecD4G4SPIRV []byte

//go:embed shaders/matvec_d4g_8.spv
var matvecD4G8SPIRV []byte

//go:embed shaders/matvec_d4g16.spv
var matvecD4G16SPIRV []byte

//go:embed shaders/matvec_d4g16_2.spv
var matvecD4G16_2SPIRV []byte

//go:embed shaders/matvec_d4g16_4.spv
var matvecD4G16_4SPIRV []byte

//go:embed shaders/matvec_d4g16_8.spv
var matvecD4G16_8SPIRV []byte

// matvecD4GNoTableSPIRV is the same kernel with the lattice lookup taken out
// and nothing put in its place. It answers one question — what the table costs
// — and is never used to compute anything.
//
//go:embed shaders/matvec_d4g_notable.spv
var matvecD4GNoTableSPIRV []byte

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

// D4GKernels is everything about the D4G product that belongs to the device
// rather than to a matrix: the lattice table, and one pipeline a pass width.
type D4GKernels struct {
	d     *Device
	bits  int
	table *Buffer
	pipes map[int]*Pipeline
}

// NewD4GKernels uploads the lattice of a code width and builds its pipelines.
func NewD4GKernels(d *Device, bits int) (*D4GKernels, error) {
	spirv, ok := d4gSPIRV(bits)
	if !ok {
		return nil, fmt.Errorf("vk: there is no D4G kernel for %d-bit codes", bits)
	}
	k := &D4GKernels{d: d, bits: bits, pipes: map[int]*Pipeline{}}
	var err error
	if k.table, err = d.Upload(packD4Table(bits)); err != nil {
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

func d4gSPIRV(bits int) (map[int][]byte, bool) {
	switch bits {
	case nn.D4Bits:
		return map[int][]byte{1: matvecD4GSPIRV, 2: matvecD4G2SPIRV, 4: matvecD4G4SPIRV, 8: matvecD4G8SPIRV}, true
	case nn.D4Bits16:
		return map[int][]byte{1: matvecD4G16SPIRV, 2: matvecD4G16_2SPIRV, 4: matvecD4G16_4SPIRV, 8: matvecD4G16_8SPIRV}, true
	}
	return nil, false
}

// Table is the lattice buffer, for a caller binding its own sets.
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

// packD4Table lays a tier's shell out as one word a point, four signed bytes.
// Every coordinate of a point of either tier fits in a byte several times over.
func packD4Table(bits int) []byte {
	pts := nn.D4TableN(bits)
	n := len(pts) / 4
	out := make([]byte, n*4)
	for i := 0; i < n; i++ {
		for j := 0; j < 4; j++ {
			out[i*4+j] = byte(int8(pts[i*4+j]))
		}
	}
	return out
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
	if cols%nn.D4Block != 0 {
		return nil, fmt.Errorf("vk: a D4G row needs a multiple of %d columns, given %d", nn.D4Block, cols)
	}
	nb := cols / nn.D4Block
	if nb%2 != 0 {
		return nil, fmt.Errorf("vk: %d columns give %d blocks a row, and the shader reads words", cols, nb)
	}
	if want := rows * nb * nn.D4BlockBytes(k.bits); len(data) != want {
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
	k    *D4GKernels
	act  *Buffer
	out  *Buffer
	pipe *Pipeline
	set  *Set
}

func newHostD4GMatrix(d *Device, data []byte, rows, cols int) (*hostD4GMatrix, error) {
	return newHostD4G(d, data, rows, cols, nil)
}

// newHostD4GMatrixNoTable builds the same from the kernel that skips the
// lookup. It computes nothing anybody wants; it prices the table.
func newHostD4GMatrixNoTable(d *Device, data []byte, rows, cols int) (*hostD4GMatrix, error) {
	return newHostD4G(d, data, rows, cols, matvecD4GNoTableSPIRV)
}

func newHostD4G(d *Device, data []byte, rows, cols int, spirv []byte) (*hostD4GMatrix, error) {
	k, err := NewD4GKernels(d, nn.D4Bits)
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
	if spirv != nil {
		// The variant is a different pipeline over the same buffers.
		if h.pipe, err = d.NewPipeline(spirv, 4, uint32(unsafe.Sizeof(d4gPush{}))); err != nil {
			h.Close()
			return nil, err
		}
		if h.set, err = h.pipe.NewSet([]*Buffer{h.weights, k.table, h.act, h.out}); err != nil {
			h.Close()
			return nil, err
		}
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
	set := h.Set(1)
	if h.set != nil {
		set = h.set
	}
	if err := set.Dispatch(h.Groups(), unsafe.Pointer(&push)); err != nil {
		return err
	}
	copy(out, h.out.Floats())
	return nil
}

func (h *hostD4GMatrix) Close() {
	if h.set != nil {
		h.set.Close()
		h.set = nil
	}
	if h.pipe != nil {
		h.pipe.Close()
		h.pipe = nil
	}
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
