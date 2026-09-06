package vk

// The golem product on the card: a trellis, decoded without a codebook.
//
// The weights go up once, as the file holds them. The step grid goes up once
// too, for the whole device — 256 floats, one copy serving every matrix of
// every model on the card. A code costs a window read, a multiply and a byte
// sum, and what it saves is a third of the bytes a Q4_K row would have cost to
// read.
//
// The activation arrives already through PrepareGolem: the per-column vector
// and the rotation belong to the site, not to the block, and both are undone on
// this side before the product rather than stored with the weights.

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// The four widths of each tier are not four copies of one kernel: golemShapes
// below says which workgroup and which decoder each of them is built with, and
// TestGolemSweep is where those came from.
// The four widths of each tier are not four copies of one kernel: golemShapes
// below says which workgroup, which decoder and which read each of them is
// built with, and TestGolemSweep is where those came from.
// One binary a tier and a width. The workgroup, the decoder and the read-ahead
// are specialization constants rather than defines, so the shape is chosen when
// the pipeline is made — see GolemShapes.
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

type golemPush struct {
	dim uint32 // outputs
	ffn uint32 // inputs
	col uint32 // the first column this dispatch answers
}

// GolemShape is how a pass width is built: the workgroup that answers it,
// whether the codebook's image lives in that workgroup's shared memory or is
// hashed again for every weight, and whether a block's stream is read one
// iteration before it is decoded. All three are specialization constants of
// vk/shaders/matvec_t4g.comp, so a shape costs a pipeline and not a binary.
type GolemShape struct {
	Threads  int  // the workgroup, eight lanes to a row
	Table    bool // the codebook's image in shared memory, or the hash per weight
	Prefetch bool // read the next block's stream while this one decodes
	Copies   int  // copies of that image, one per group of lanes, against bank conflicts
}

// Spec is the three constants in the order the shader numbers them.
func (s GolemShape) Spec() []uint32 {
	b := func(v bool) uint32 {
		if v {
			return 1
		}
		return 0
	}
	copies := s.Copies
	if copies < 1 {
		copies = 1
	}
	return []uint32{uint32(s.Threads), b(s.Table), b(s.Prefetch), uint32(copies)}
}

func (s GolemShape) String() string {
	copies := s.Copies
	if copies < 1 {
		copies = 1
	}
	return fmt.Sprintf("%d,%t,%t,%d", s.Threads, s.Table, s.Prefetch, copies)
}

// golemDefaultShapes is what cmd/golemtune measured on an RX 9070 XT (RADV,
// RDNA4): every shape interleaved in one process, the order rotating between
// rounds, the fastest of twelve, on a 9728x2560 product. Microseconds:
//
//	width  hash/128  hash/256  table/128  table/256  table/512  +prefetch 128/256/512
//	    1      49.6      49.2       46.0      44.4       44.5     44.5 / 42.4 / 44.8
//	    2      57.9      57.2       56.1      61.2       54.6     55.9 / 60.9 / 55.2
//	    4      79.5      79.6       80.8      99.3       84.5     83.2 / 96.4 / 95.6
//	    8     138.7     138.0      137.3     164.8      167.6    139.7 /187.4 /188.6
//
// A narrow pass decodes a weight for one activation and wants the table; a pass
// of eight spends each decoded weight eight times, so the hash is already
// amortized and all the table has left to offer is sixteen kibibytes against
// the occupancy. The read-ahead is worth having only at one column, where the
// wave has nothing else to wait behind.
//
// The workgroup is not one number either, and an earlier version of this table
// said it was. Measured without rotating the order — which is worth three to
// six per cent by itself, see vk/golem_bar_test.go — the two-column row came
// out ten per cent wrong; four and eight are ties between 128 and 256 threads
// that change places between runs, and take 256 for being the others' number.
var golemDefaultShapes = map[int]GolemShape{
	1: {Threads: 256, Table: true, Prefetch: true, Copies: 1},
	2: {Threads: 512, Table: true, Prefetch: false, Copies: 1},
	4: {Threads: 256, Table: false, Prefetch: false, Copies: 1},
	8: {Threads: 256, Table: false, Prefetch: false, Copies: 1},
}

// The shape of the trade above follows from the pass width and would come out
// the same way on any card. Where the crossing falls, and which workgroup wins,
// is the card's and the matrix's answer together, so none of it is compiled in:
// run cmd/golemtune — at the geometry of a matrix the model actually has, with
// -rows and -cols — and set GOLEM_MATVEC_SHAPE to what it prints. Every shape
// answers the same numbers whatever is fastest, and TestGolemBuildsAgree holds
// them to it.
//
// And read what it prints as a candidate rather than an answer. golemtune times
// one matrix in a loop, where a model reads sixty of them with everything else
// a block does between; the two disagree, and they disagree in both directions.
// Measured on Qwen3.8-27B in t3g, at its own 17408x5120 geometry, against
// qwen35's TestVulkanWidthCost which times the same shapes inside a pass:
//
//	width  golemtune says   in a pass   what a pass measured
//	    1  256,true,true    prefetch off      27.7 -> 27.1 ms
//	    2  512,true,true    256,true,false    31.8 -> 30.2 ms
//
// The table above is Qwen3-4B's and the 4B still wants what it says — its t4g
// draws at 75.6 tokens a second with the read-ahead and 76.9 without, which is
// inside the noise, and the fastest single pass of either belongs to the
// read-ahead. So the compiled default stays the 4B's and the 27B is a
// GOLEM_MATVEC_SHAPE away, which is what the setting is for. What is worth
// carrying away is that an isolated timing is a hypothesis: it has been wrong
// about the mat-vec's shape, about the K-quant kernel's, and about this.

// GolemShapes is the shape each pass width is built with. It is
// golemDefaultShapes unless GOLEM_MATVEC_SHAPE says otherwise, in the form
// cmd/golemtune prints: "1:256,true,true 2:128,true,true 4:128,true,false
// 8:256,false,false", widths it does not name keeping their default.
//
// A malformed setting is a mistake worth hearing about rather than working
// around, so it is reported once on the standard error and then ignored.
func GolemShapes() map[int]GolemShape {
	golemShapesOnce.Do(func() {
		golemShapes = map[int]GolemShape{}
		for w, s := range golemDefaultShapes {
			golemShapes[w] = s
		}
		spec := os.Getenv("GOLEM_MATVEC_SHAPE")
		if spec == "" {
			return
		}
		for _, field := range strings.Fields(spec) {
			w, s, err := parseGolemShape(field)
			if err != nil {
				fmt.Fprintf(os.Stderr, "vk: GOLEM_MATVEC_SHAPE %q: %v\n", field, err)
				continue
			}
			golemShapes[w] = s
		}
	})
	return golemShapes
}

var (
	golemShapesOnce sync.Once
	golemShapes     map[int]GolemShape
)

// parseGolemShape reads one "width:threads,table,prefetch" field.
func parseGolemShape(field string) (int, GolemShape, error) {
	width, rest, ok := strings.Cut(field, ":")
	if !ok {
		return 0, GolemShape{}, fmt.Errorf("want width:threads,table,prefetch")
	}
	w, err := strconv.Atoi(width)
	if err != nil {
		return 0, GolemShape{}, fmt.Errorf("width: %w", err)
	}
	if _, ok := golemDefaultShapes[w]; !ok {
		return 0, GolemShape{}, fmt.Errorf("there is no pass of %d columns", w)
	}
	parts := strings.Split(rest, ",")
	if len(parts) != 3 && len(parts) != 4 {
		return 0, GolemShape{}, fmt.Errorf("want threads,table,prefetch[,copies]")
	}
	var s GolemShape
	if s.Threads, err = strconv.Atoi(parts[0]); err != nil {
		return 0, GolemShape{}, fmt.Errorf("threads: %w", err)
	}
	// Eight lanes to a row, and a workgroup the card will take.
	if s.Threads%8 != 0 || s.Threads < 8 || s.Threads > 1024 {
		return 0, GolemShape{}, fmt.Errorf("threads: %d is not eight lanes a row up to 1024", s.Threads)
	}
	if s.Table, err = strconv.ParseBool(parts[1]); err != nil {
		return 0, GolemShape{}, fmt.Errorf("table: %w", err)
	}
	if s.Prefetch, err = strconv.ParseBool(parts[2]); err != nil {
		return 0, GolemShape{}, fmt.Errorf("prefetch: %w", err)
	}
	s.Copies = 1
	if len(parts) == 4 {
		if s.Copies, err = strconv.Atoi(parts[3]); err != nil {
			return 0, GolemShape{}, fmt.Errorf("copies: %w", err)
		}
		// Sixteen kibibytes a copy, and a workgroup has sixty-four.
		if s.Copies < 1 || s.Copies > 4 {
			return 0, GolemShape{}, fmt.Errorf("copies: %d is not one to four", s.Copies)
		}
	}
	return w, s, nil
}

// golemRowsPerGroup is the shader's OUTS for a pass of that width: eight lanes
// to a row, so a workgroup answers an eighth of itself.
func golemRowsPerGroup(columns int) int { return GolemShapes()[columns].Threads / 8 }

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
	if k.table, err = d.Upload(golemTable()); err != nil {
		return nil, err
	}
	for _, columns := range GolemWidths {
		p, err := d.NewPipelineSpec(spirv[columns], 4, golemPushSize, GolemShapes()[columns].Spec())
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
// only table the card is handed. The codebook is not in it: the trellis decodes
// by arithmetic, and vk/shaders/matvec_t4g.comp builds the image of that
// arithmetic in its own shared memory once a workgroup.
func golemTable() []byte {
	out := make([]byte, 256*4)
	for c := 0; c < 256; c++ {
		binary.LittleEndian.PutUint32(out[c*4:], math.Float32bits(nn.T4GStep(byte(c))))
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
	unit := nn.T4GSeq
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
		per := golemRowsPerGroup(columns)
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
