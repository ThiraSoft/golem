package vk

// The tiled product, as a thing that can be tested on its own.
//
// shaders/matmul.comp says what it is and why a prompt wants it. This file is
// the smallest wrapper that lets a test hand it one matrix and a batch of
// activations and read the answer back; the stack binds the same shader to its
// own buffers and never comes through here.

import (
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O -DCOLUMNS=32 -DBN=32 -DBM=64 -DBK=128 -DWAVE_M=4 -DWAVE_N=1 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop32.spv
//go:generate glslc -O -DCOLUMNS=64 -DBN=64 -DBM=128 -DBK=128 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop64.spv
//go:generate glslc -O -DCOLUMNS=128 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop128.spv
//go:generate glslc -O -DCOLUMNS=256 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop256.spv
//go:generate glslc -O -DCOLUMNS=512 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop512.spv
//go:generate glslc -O -DQ5K -DCOLUMNS=64 -DBN=64 -DBM=128 -DBK=128 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q5k64.spv
//go:generate glslc -O -DQ5K -DCOLUMNS=128 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q5k128.spv
//go:generate glslc -O -DQ5K -DCOLUMNS=256 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q5k256.spv
//go:generate glslc -O -DQ5K -DCOLUMNS=512 -DBN=128 -DBM=128 -DBK=32 -DWAVE_M=2 -DWAVE_N=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_q5k512.spv
//go:generate glslc -O -DCOLUMNS=64 -DBN=64 -DBM=128 -DBK=64 -DWAVE_M=2 -DWAVE_N=2 -DBYID --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop_id.spv
//go:generate glslc -O -DCOLUMNS=32 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul32.spv
//go:generate glslc -O -DCOLUMNS=64 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul64.spv
//go:generate glslc -O -DCOLUMNS=128 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul128.spv
//go:generate glslc -O -DCOLUMNS=256 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul256.spv
//go:generate glslc -O -DCOLUMNS=512 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul512.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul8.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_reduce.comp -o shaders/matmul_reduce.spv

// The engine binds the tiled product itself; this wrapper is the bench's and
// the parity test's. The cooperative product is what nothing binds yet, and
// the header of shaders/matmul_coop.comp says why.

// The widths the tiled product is built at. shaders/matmul.comp gives a thread
// four rows by four columns, so a pass narrower than four columns has nothing
// to tile and the mat-vec kernel is the right shape for it.
const (
	wideColumns  = 512
	tiledColumns = 32
	smallColumns = 8
)

// matmulWidths are every width a pass may run at, smallest first. A stretch
// takes the narrowest that holds it: a prompt of sixty-four read by the
// hundred-and-twenty-eight column binary computes sixty-four columns nobody
// asked for, and measured that is a third of the rate the right binary gives
// it.
var matmulWidths = []int{1, smallColumns, tiledColumns, 64, 128, 256, wideColumns}

// matmulRows is shaders/matmul.comp's BM: how many rows one workgroup writes.
const matmulRows = 32

// matmulColBlock is shaders/matmul.comp's BN: how many columns of the batch
// one workgroup answers. Above it a pass is cut across its columns too — the
// activation staged for a step is BN*(KSTEP*8+1) uints, and at sixty-four
// columns that is already more shared memory than a workgroup may have.
const matmulColBlock = 32

// matmulColGroups is how many of those a pass of that width makes.
func matmulColGroups(width int) int {
	if width <= matmulColBlock {
		return 1
	}
	return width / matmulColBlock
}

// matmulSplit is how many slices of the shared dimension the tiled product
// cuts a matrix of that many rows into.
//
// A product writes matmulRows rows to a workgroup, so a matrix with few rows
// gives few workgroups: the Qwen3 4B's ffn_down gives eighty against
// sixty-four compute units, and the card is idle for want of anything to run.
// Cutting the shared dimension multiplies them and costs one pass of
// shaders/matmul_reduce.comp over the answer. Measured on that matrix at
// thirty-two columns, 4.24 microseconds a column against 3.65 — and on
// ffn_gate, which has three hundred and four workgroups and needs none of it,
// the same split is a fifth slower. So it is the row count that decides.
func matmulSplit(rows int) int {
	if groups := (rows + matmulRows - 1) / matmulRows; groups <= 96 {
		return matmulSlices
	}
	return 1
}

// matmulSlices is the split a row-poor product gets. Three and four measure
// the same and five is worse.
const matmulSlices = 4

// matmulCoopRows is shaders/matmul_coop.comp's BM.
func matmulCoopRows(width int) int {
	if width <= tiledColumns {
		return 64
	}
	return 128
}

// matmulCoopColBlock is that kernel's BN, and has to agree with the -DBN the
// generate lines above pass it.
func matmulCoopColBlock(width int) int {
	if width <= tiledColumns {
		return 32
	}
	if width <= 64 {
		return 64
	}
	return 128
}

// coopSplit is the slicing of the shared dimension a matrix of that many rows
// takes. The cooperative product's tile is wider than the integer one's, so a
// row-poor matrix leaves it fewer workgroups and wants the split; a row-rich
// one already fills the card and the split only costs it a reduction.
//
// Every caller has to agree on this, and on matmulCoopRows with it: the number
// here and the -DBM the generate lines pass are the same number said twice,
// and left disagreeing the dispatch covers a fraction of the answer and leaves
// the rest zero. It reads as a speed-up.
func coopSplit(rows, columns int, coop bool) int {
	if coop && rows/matmulCoopRows(columns) > 96 {
		return 1
	}
	return matmulSplit(rows)
}

// coopProductGroups is the workgroup count a pass of that width dispatches to
// answer that many outputs, for whichever product is bound at that width.
func coopProductGroups(coop bool, width, outputs int) uint32 {
	if width < tiledColumns {
		return uint32((outputs + matvecOuts - 1) / matvecOuts)
	}
	rows, cols := matmulRows, matmulColGroups(width)
	if coop {
		rows, cols = matmulCoopRows(width), matmulCoopColGroups(width)
	}
	return uint32((outputs+rows-1)/rows) * uint32(cols)
}

// matmulCoopColGroups is how many of those a pass of that width makes.
func matmulCoopColGroups(width int) int {
	block := matmulCoopColBlock(width)
	if width <= block {
		return 1
	}
	return width / block
}

// A MatMul is one Q4_0 matrix resident on a device with the buffers a batch
// passes through.
type MatMul struct {
	d          *Device
	rows, cols int
	columns    int
	// perGroup is how many rows one workgroup of the chosen kernel writes.
	perGroup int

	weights, aq, as, out *Buffer
	// back is where Run copies the answer to read it, and out is device
	// memory. A shader that writes its answer straight into a host-visible
	// buffer writes it across the bus: about twelve gigabytes a second on this
	// card against the hundreds VRAM gives. The benchmark used to allocate out
	// that way and so measured PCIe rather than the kernel — and it measured
	// it against llama.cpp's twin, which has always written to device memory
	// and copied afterwards. See the header of shaders/matmul_coop.comp.
	back *Buffer
	// split is how many slices of the shared dimension the product is cut
	// into, parts is where the slices land, and reduce adds them into out.
	split      int
	parts      *Buffer
	reducePipe *Pipeline
	reduceSet  *Set
	// The activation is staged: written here and copied into device memory
	// before a pass. It used to be read straight out of system memory, and a
	// benchmark built that way measures the bus rather than the kernel —
	// vk/device.go's Host says what that was worth in the engine.
	stageQ, stageS *Buffer
	coop           bool
	pipe           *Pipeline
	set            *Set

	byID                               bool
	experts                            int
	counts, pairs, plan                *Buffer
	stageCounts, stagePairs, stagePlan *Buffer
}

// NewMatMul uploads a Q4_0 matrix in the file's own layout and binds the tiled
// product to it, for passes of the given width.
func NewMatMul(d *Device, data []byte, rows, cols, columns int, coop bool) (*MatMul, error) {
	if cols%nn.QuantBlock != 0 {
		return nil, fmt.Errorf("vk: a Q4_0 row needs a multiple of %d columns, given %d", nn.QuantBlock, cols)
	}
	if want := rows * rowBytesQ4_0(cols); len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns need %d bytes, given %d", rows, cols, want, len(data))
	}
	spirv, err := matmulSPIRV(columns)
	if err != nil && !coop {
		return nil, err
	}
	perGroup, wave := matmulRows, uint32(0)
	if coop && d.Coopmat() {
		wave = coopmatWave
		if rows%coopTile != 0 {
			return nil, fmt.Errorf("vk: the cooperative product writes %d rows at a time, and %d is not a multiple of it", coopTile, rows)
		}
		if spirv, err = matmulCoopSPIRV(columns); err != nil {
			return nil, err
		}
		perGroup = matmulCoopRows(columns)
	}

	// The cooperative product takes the same split as the integer one: its
	// tile is wider, so a row-poor matrix leaves it even fewer workgroups than
	// shaders/matmul.comp's, and its own header says the split and the column
	// blocking together are what first put it ahead of the dot products.
	split := coopSplit(rows, columns, coop)
	m := &MatMul{d: d, rows: rows, cols: cols, columns: columns, perGroup: perGroup, split: split, coop: coop && d.Coopmat()}
	layout := splitQ4_0(data, rows, cols)
	if m.weights, err = d.Upload(layout); err != nil {
		return nil, err
	}
	nb := cols / nn.QuantBlock
	for _, spec := range []struct {
		into **Buffer
		size int
		back bool
	}{
		{&m.stageQ, cols * columns, false},
		{&m.stageS, 2 * nb * 4 * columns, false},
		{&m.aq, cols * columns, false},
		{&m.as, 2 * nb * 4 * columns, false},
		{&m.out, rows * 4 * columns, false},
		{&m.back, rows * 4 * columns, true},
	} {
		var b *Buffer
		switch {
		case spec.back:
			b, err = d.Readback(spec.size, bufferUsageTransferDst)
		case spec.into == &m.stageQ || spec.into == &m.stageS:
			b, err = d.Host(spec.size, bufferUsageTransferSrc)
		case spec.into == &m.out:
			b, err = d.Local(spec.size, bufferUsageStorage|bufferUsageTransferSrc)
		default:
			b, err = d.Local(spec.size, bufferUsageStorage|bufferUsageTransferDst)
		}
		if err != nil {
			m.Close()
			return nil, err
		}
		*spec.into = b
	}

	target := m.out
	if m.split > 1 {
		if m.parts, err = d.Local(rows*4*columns*m.split, bufferUsageStorage); err != nil {
			m.Close()
			return nil, err
		}
		target = m.parts
		if m.reducePipe, err = d.newPipeline(matmulReduceSPIRV, 2, uint32(unsafe.Sizeof(moePush{})), 0, nil); err != nil {
			m.Close()
			return nil, err
		}
		if m.reduceSet, err = m.reducePipe.NewSet([]*Buffer{m.parts, m.out}); err != nil {
			m.Close()
			return nil, err
		}
	}
	buffers := []*Buffer{m.weights, m.aq, m.as, target}
	if m.pipe, err = d.newPipeline(spirv, len(buffers), uint32(unsafe.Sizeof(moePush{})), wave, nil); err != nil {
		m.Close()
		return nil, err
	}
	if m.set, err = m.pipe.NewSet(buffers); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

// NewMatMulByID is NewMatMul reading shaders/moe_scatter.comp's lists: the
// weights are a stack of `experts` slabs and a workgroup answers the entries
// of one expert's list. Bindings 4, 5 and 6 are counts, pairs and plan, in
// that order, which is the order shaders/matmul.comp's BYID build already
// takes and the order vk/mixture.go binds.
func NewMatMulByID(d *Device, stack []byte, rows, cols, experts, columns int) (*MatMul, error) {
	if cols%nn.QuantBlock != 0 {
		return nil, fmt.Errorf("vk: a Q4_0 row needs a multiple of %d columns, given %d", nn.QuantBlock, cols)
	}
	want := experts * rows * rowBytesQ4_0(cols)
	if len(stack) != want {
		return nil, fmt.Errorf("vk: %d experts of %d rows by %d columns need %d bytes, given %d", experts, rows, cols, want, len(stack))
	}
	coop := d.Coopmat()
	spirv, wave, perGroup := matmulByIDSPIRV, uint32(0), matmulRows
	if coop {
		spirv, wave, perGroup = matmulCoopByIDSPIRV, coopmatWave, idProductCoopBM
	}
	m := &MatMul{d: d, rows: rows, cols: cols, columns: columns, experts: experts, perGroup: perGroup, coop: coop, byID: true}
	layout := splitQ4_0(stack, experts*rows, cols)
	var err error
	if m.weights, err = d.Upload(layout); err != nil {
		return nil, err
	}
	nb := cols / nn.QuantBlock
	pairsMax := columns * expertsUsed
	for _, spec := range []struct {
		into **Buffer
		size int
		back bool
	}{
		{&m.stageQ, cols * pairsMax, false},
		{&m.stageS, 2 * nb * 4 * pairsMax, false},
		{&m.stageCounts, experts * 4, false},
		{&m.stagePairs, experts * columns * 4, false},
		{&m.stagePlan, (1 + 2*idPlanMax(experts, columns, idProductBN)) * 4, false},
		{&m.aq, cols * pairsMax, false},
		{&m.as, 2 * nb * 4 * pairsMax, false},
		{&m.out, rows * 4 * pairsMax, false},
		{&m.back, rows * 4 * pairsMax, true},
		{&m.counts, experts * 4, false},
		{&m.pairs, experts * columns * 4, false},
		{&m.plan, (1 + 2*idPlanMax(experts, columns, idProductBN)) * 4, false},
	} {
		var b *Buffer
		switch {
		case spec.back:
			b, err = d.Readback(spec.size, bufferUsageTransferDst)
		case spec.into == &m.stageQ || spec.into == &m.stageS || spec.into == &m.stageCounts || spec.into == &m.stagePairs || spec.into == &m.stagePlan:
			b, err = d.Host(spec.size, bufferUsageTransferSrc)
		case spec.into == &m.out:
			b, err = d.Local(spec.size, bufferUsageStorage|bufferUsageTransferSrc)
		default:
			b, err = d.Local(spec.size, bufferUsageStorage|bufferUsageTransferDst)
		}
		if err != nil {
			m.Close()
			return nil, err
		}
		*spec.into = b
	}
	buffers := []*Buffer{m.weights, m.aq, m.as, m.out, m.counts, m.pairs, m.plan}
	push := uint32(unsafe.Sizeof(moePush{}))
	if m.pipe, err = d.newPipeline(spirv, len(buffers), push, wave, nil); err != nil {
		m.Close()
		return nil, err
	}
	if m.set, err = m.pipe.NewSet(buffers); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

func (m *MatMul) SetCounts(counts []uint32) {
	copy(m.stageCounts.Bytes(), asBytesUint32(counts))
}

func (m *MatMul) SetPairs(pairs []uint32) {
	copy(m.stagePairs.Bytes(), asBytesUint32(pairs))
}

func (m *MatMul) SetPlan(plan []uint32) {
	copy(m.stagePlan.Bytes(), asBytesUint32(plan))
}

func (m *MatMul) SetIDPairColumn(pair int, b *nn.Batch, used int) error {
	if b.Q == nil {
		return fmt.Errorf("vk: a Q4_0 product needs the activation in its Q8_0 form")
	}
	nb := m.cols / nn.QuantBlock
	dst := m.stageQ.Bytes()[pair*m.cols:]
	for i, v := range b.Q[:m.cols] {
		dst[i] = byte(v)
	}
	scbase := (pair/used)*2*used*nb + (pair%used)*nb
	scales := m.stageS.Floats()[scbase:]
	copy(scales[:nb], b.Scales[:nb])
	corr := m.stageS.Floats()[scbase+used*nb:]
	copy(corr[:nb], b.Corr[:nb])
	return nil
}

func (m *MatMul) RunByID(out [][]float32, used int) error {
	if err := m.d.Submit(func(r *Recorder) {
		m.upload(r)
		push := moePush{dim: uint32(m.rows), ffn: uint32(m.cols), used: uint32(used), cap: uint32(m.columns), split: 1}
		rows := uint32((m.rows + m.perGroup - 1) / m.perGroup)
		r.Dispatch(m.set, rows*uint32(idPlanMax(m.experts, m.columns, idBN(m.coop))), unsafe.Pointer(&push))
		r.Barrier()
		r.CopyFrom(m.back, 0, m.out, 0, len(m.back.Bytes()))
	}); err != nil {
		return err
	}
	answer := m.back.Floats()
	for p := range out {
		copy(out[p], answer[p*m.rows:(p+1)*m.rows])
	}
	return nil
}

func asBytesUint32(u []uint32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&u[0])), len(u)*4)
}

// matmulWide is the tiled product built at the width of a pass, which is a
// constant: a width the generate lines do not build is a build failure here
// rather than a run-time one.
func matmulWide() []byte {
	spirv, err := matmulSPIRV(wideColumns)
	if err != nil {
		panic(err)
	}
	return spirv
}

// matmulCoopSPIRV is the cooperative product built for that many columns.
func matmulCoopSPIRV(columns int) ([]byte, error) {
	switch columns {
	case 32:
		return matmulCoop32SPIRV, nil
	case 64:
		return matmulCoop64SPIRV, nil
	case 128:
		return matmulCoop128SPIRV, nil
	case 256:
		return matmulCoop256SPIRV, nil
	case 512:
		return matmulCoop512SPIRV, nil
	}
	return nil, fmt.Errorf("vk: the cooperative product is built at 32, 64, 128, 256, 512 columns, not %d", columns)
}

// matmulSPIRV is the binary built for that many columns.
func matmulSPIRV(columns int) ([]byte, error) {
	switch columns {
	case 512:
		return matmulWidest512SPIRV, nil
	case 256:
		return matmulWidest256SPIRV, nil
	case 128:
		return matmulWidest128SPIRV, nil
	case 64:
		return matmulWide64SPIRV, nil
	case 32:
		return matmulWide32SPIRV, nil
	case smallColumns:
		return matmulSmallSPIRV, nil
	}
	return nil, fmt.Errorf("vk: the tiled product is built at %d, 32, 64, 128, 256, 512 columns, not %d",
		smallColumns, columns)
}

// SetColumn writes one column of the batch, which must already carry its Q8_0
// form. A batch of one, for the reason Q40Head.MatVec gives: nn.Batch
// interleaves by block and then by column, so a column is contiguous only when
// there is one.
func (m *MatMul) SetColumn(column int, b *nn.Batch) error {
	if b.Q == nil {
		return fmt.Errorf("vk: a Q4_0 product needs the activation in its Q8_0 form")
	}
	if b.Size != 1 || b.Width != m.cols {
		return fmt.Errorf("vk: a column is one activation of %d, given %d of %d", m.cols, b.Size, b.Width)
	}
	if column < 0 || column >= m.columns {
		return fmt.Errorf("vk: column %d of %d", column, m.columns)
	}
	nb := m.cols / nn.QuantBlock
	dst := m.stageQ.Bytes()[column*m.cols:]
	for i, v := range b.Q[:m.cols] {
		dst[i] = byte(v)
	}
	scales := m.stageS.Floats()[column*2*nb:]
	copy(scales[:nb], b.Scales[:nb])
	copy(scales[nb:2*nb], b.Corr[:nb])
	return nil
}

// Run computes the whole batch and writes one answer per column.
func (m *MatMul) Run(out [][]float32) error {
	if len(out) != m.columns {
		return fmt.Errorf("vk: the pass writes %d columns, given %d", m.columns, len(out))
	}
	for _, o := range out {
		if len(o) != m.rows {
			return fmt.Errorf("vk: a column writes %d outputs, given %d", m.rows, len(o))
		}
	}
	if err := m.d.Submit(func(r *Recorder) {
		m.upload(r)
		m.pass(r)
		r.Barrier()
		r.CopyFrom(m.back, 0, m.out, 0, m.rows*4*m.columns)
	}); err != nil {
		return err
	}
	answer := m.back.Floats()
	for c := range out {
		copy(out[c], answer[c*m.rows:])
	}
	return nil
}

// RunTimes repeats the pass n times inside a single submission, with the
// inputs already in place. It measures the kernel rather than the round trip:
// a submission costs sixty-three microseconds whatever is in it, and a card
// handed one pass and then left alone runs at half its clocks. Mixture.RunTimes
// says the same about the feed forward.
func (m *MatMul) RunTimes(n int) error {
	return m.d.Submit(func(r *Recorder) {
		m.upload(r)
		for i := 0; i < n; i++ {
			if i > 0 {
				r.Barrier()
			}
			m.pass(r)
		}
	})
}

// pass is the product, and the reduction after it when the shared dimension
// was split.
func (m *MatMul) pass(r *Recorder) {
	push := moePush{dim: uint32(m.rows), ffn: uint32(m.cols), used: 1, split: uint32(m.split)}
	cols := matmulColGroups(m.columns)
	if m.coop {
		cols = matmulCoopColGroups(m.columns)
	}
	groups := uint32((m.rows+m.perGroup-1)/m.perGroup) * uint32(cols) * uint32(m.split)
	r.Dispatch(m.set, groups, unsafe.Pointer(&push))
	if m.split > 1 {
		r.Barrier()
		sum := moePush{dim: uint32(m.rows), ffn: uint32(m.columns), used: uint32(m.split)}
		r.Dispatch(m.reduceSet, uint32((m.rows*m.columns+255)/256), unsafe.Pointer(&sum))
	}
}

// upload puts the staged activation into device memory. It is one copy of a
// few hundred kilobytes at the head of a submission, and the sixty-four passes
// after it read it from where the engine's own kernels would have left it.
func (m *MatMul) upload(r *Recorder) {
	r.CopyFrom(m.aq, 0, m.stageQ, 0, len(m.stageQ.Bytes()))
	r.CopyFrom(m.as, 0, m.stageS, 0, len(m.stageS.Bytes()))
	if m.byID {
		r.CopyFrom(m.counts, 0, m.stageCounts, 0, len(m.stageCounts.Bytes()))
		r.CopyFrom(m.pairs, 0, m.stagePairs, 0, len(m.stagePairs.Bytes()))
		r.CopyFrom(m.plan, 0, m.stagePlan, 0, len(m.stagePlan.Bytes()))
	}
	r.Barrier()
}

func (m *MatMul) Close() {
	if m.set != nil {
		m.set.Close()
		m.set = nil
	}
	if m.pipe != nil {
		m.pipe.Close()
		m.pipe = nil
	}
	if m.reduceSet != nil {
		m.reduceSet.Close()
		m.reduceSet = nil
	}
	if m.reducePipe != nil {
		m.reducePipe.Close()
		m.reducePipe = nil
	}
	for _, b := range []**Buffer{&m.parts, &m.back, &m.out, &m.as, &m.aq, &m.stageS, &m.stageQ, &m.weights, &m.counts, &m.pairs, &m.plan, &m.stageCounts, &m.stagePairs, &m.stagePlan} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}
