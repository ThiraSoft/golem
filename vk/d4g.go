package vk

// The D4G product on the card.
//
// The weights go up once, as the file holds them. The lattice goes up once too
// — sixteen kibibytes for the whole model, because the table is the lattice and
// not a codebook fitted to a tensor — and it is the reason this format is
// worth a kernel at all: a code costs one lookup and four sign extensions, and
// what it saves is a third of the bytes a Q4_K row would have cost to read.
//
// The activation arrives already through nn.PrepareD4G: the per-column vector
// and the rotation belong to the matrix, not to the block, and both are undone
// on this side before the product rather than stored with the weights.

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g.spv

//go:embed shaders/matvec_d4g.spv
var matvecD4GSPIRV []byte

//go:generate glslc -O -DNOTABLE --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_d4g.comp -o shaders/matvec_d4g_notable.spv

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

// D4GMatrix is one D4G weight matrix resident on the device.
type D4GMatrix struct {
	d          *Device
	rows, cols int

	weights *Buffer
	table   *Buffer
	act     *Buffer
	out     *Buffer

	pipe   *Pipeline
	set    *Set
	groups uint32

	// noTable builds the pipeline from the variant that does not read the
	// lattice. Only the benchmark sets it.
	noTable bool
}

// d4gRowsPerGroup is the shader's OUTS: a workgroup of 128 answers sixteen
// rows, eight lanes to a row.
const d4gRowsPerGroup = 16

// NewD4GMatrix uploads a matrix in the form the file holds it — per row, the
// plane of fp16 steps then the plane of twelve-bit codes.
func NewD4GMatrix(d *Device, data []byte, rows, cols int) (*D4GMatrix, error) {
	return newD4GMatrix(d, data, rows, cols, false)
}

// newD4GMatrixNoTable is the same, built from the kernel that skips the lookup.
// It computes nothing anybody wants; it prices the table.
func newD4GMatrixNoTable(d *Device, data []byte, rows, cols int) (*D4GMatrix, error) {
	return newD4GMatrix(d, data, rows, cols, true)
}

func newD4GMatrix(d *Device, data []byte, rows, cols int, noTable bool) (*D4GMatrix, error) {
	if cols%nn.D4Block != 0 {
		return nil, fmt.Errorf("vk: a D4G row needs a multiple of %d columns, given %d", nn.D4Block, cols)
	}
	nb := cols / nn.D4Block
	if nb%2 != 0 {
		return nil, fmt.Errorf("vk: %d columns give %d blocks a row, and the shader reads words", cols, nb)
	}
	if want := rows * nb * 26; len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns need %d bytes, given %d", rows, cols, want, len(data))
	}

	m := &D4GMatrix{d: d, rows: rows, cols: cols}
	m.noTable = noTable
	var err error
	if m.weights, err = d.Upload(data); err != nil {
		return nil, err
	}
	if m.table, err = d.Upload(packD4Table()); err != nil {
		m.Close()
		return nil, err
	}
	if m.act, err = d.Host(cols*4, bufferUsageStorage); err != nil {
		m.Close()
		return nil, err
	}
	if m.out, err = d.Readback(rows*4, bufferUsageStorage); err != nil {
		m.Close()
		return nil, err
	}

	buffers := []*Buffer{m.weights, m.table, m.act, m.out}
	spirv := matvecD4GSPIRV
	if m.noTable {
		spirv = matvecD4GNoTableSPIRV
	}
	if m.pipe, err = d.NewPipeline(spirv, len(buffers), uint32(unsafe.Sizeof(d4gPush{}))); err != nil {
		m.Close()
		return nil, err
	}
	if m.set, err = m.pipe.NewSet(buffers); err != nil {
		m.Close()
		return nil, err
	}
	m.groups = uint32((rows + d4gRowsPerGroup - 1) / d4gRowsPerGroup)
	return m, nil
}

// packD4Table lays the shell out as one word a point, four signed bytes. Every
// coordinate of a point within a squared radius of forty fits in a byte several
// times over.
func packD4Table() []byte {
	pts := nn.D4Table()
	out := make([]byte, nn.D4Points()*4)
	for i := 0; i < nn.D4Points(); i++ {
		for j := 0; j < 4; j++ {
			out[i*4+j] = byte(int8(pts[i*4+j]))
		}
	}
	return out
}

// MatVec computes y = W*x for one activation, which must already have been
// through nn.PrepareD4G.
func (m *D4GMatrix) MatVec(x []float32, out []float32) error {
	if len(x) != m.cols {
		return fmt.Errorf("vk: the matrix reads %d inputs, given %d", m.cols, len(x))
	}
	if len(out) != m.rows {
		return fmt.Errorf("vk: the matrix writes %d outputs, given %d", m.rows, len(out))
	}
	copy(m.act.Floats(), x)
	push := d4gPush{dim: uint32(m.rows), ffn: uint32(m.cols), col: 0}
	if err := m.set.Dispatch(m.groups, unsafe.Pointer(&push)); err != nil {
		return err
	}
	copy(out, m.out.Floats())
	return nil
}

func (m *D4GMatrix) Close() {
	if m.set != nil {
		m.set.Close()
	}
	if m.pipe != nil {
		m.pipe.Close()
	}
	for _, b := range []*Buffer{m.weights, m.table, m.act, m.out} {
		if b != nil {
			b.Close()
		}
	}
}
