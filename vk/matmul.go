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

//go:generate glslc -O -DCOLUMNS=32 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop32.spv
//go:generate glslc -O -DCOLUMNS=64 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop64.spv
//go:generate glslc -O -DCOLUMNS=128 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop128.spv
//go:generate glslc -O -DCOLUMNS=256 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_coop.comp -o shaders/matmul_coop256.spv
//go:generate glslc -O -DCOLUMNS=32 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul32.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul8.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_reduce.comp -o shaders/matmul_reduce.spv

// The engine binds the tiled product itself; this wrapper is the bench's and
// the parity test's. The cooperative product is what nothing binds yet, and
// the header of shaders/matmul_coop.comp says why.

// The widths the tiled product is built at. shaders/matmul.comp gives a thread
// four rows by four columns, so a pass narrower than four columns has nothing
// to tile and the mat-vec kernel is the right shape for it.
const (
	wideColumns  = 32
	smallColumns = 8
)

// matmulRows is shaders/matmul.comp's BM: how many rows one workgroup writes.
const matmulRows = 32

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

// matmulCoopRows is shaders/matmul_coop.comp's BM, which is its own number:
// the cooperative product works in sixteen-row blocks and takes four of them.
const matmulCoopRows = 64

// A MatMul is one Q4_0 matrix resident on a device with the buffers a batch
// passes through.
type MatMul struct {
	d          *Device
	rows, cols int
	columns    int
	// perGroup is how many rows one workgroup of the chosen kernel writes.
	perGroup int

	weights, aq, as, out *Buffer
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
		perGroup = matmulCoopRows
	}

	split := matmulSplit(rows)
	if coop {
		split = 1 // the cooperative product is not bound in and not split
	}
	m := &MatMul{d: d, rows: rows, cols: cols, columns: columns, perGroup: perGroup, split: split, coop: coop && d.Coopmat()}
	layout := splitQ4_0(data, rows, cols)
	if coop && d.Coopmat() {
		layout = tileQ4_0(data, rows, cols, perGroup)
	}
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
		{&m.out, rows * 4 * columns, true},
	} {
		var b *Buffer
		switch {
		case spec.back:
			b, err = d.Readback(spec.size, bufferUsageStorage)
		case spec.into == &m.stageQ || spec.into == &m.stageS:
			b, err = d.Host(spec.size, bufferUsageTransferSrc)
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
		if m.reducePipe, err = d.newPipeline(matmulReduceSPIRV, 2, uint32(unsafe.Sizeof(moePush{})), 0); err != nil {
			m.Close()
			return nil, err
		}
		if m.reduceSet, err = m.reducePipe.NewSet([]*Buffer{m.parts, m.out}); err != nil {
			m.Close()
			return nil, err
		}
	}
	buffers := []*Buffer{m.weights, m.aq, m.as, target}
	if m.pipe, err = d.newPipeline(spirv, len(buffers), uint32(unsafe.Sizeof(moePush{})), wave); err != nil {
		m.Close()
		return nil, err
	}
	if m.set, err = m.pipe.NewSet(buffers); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
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
	}
	return nil, fmt.Errorf("vk: the cooperative product is built at 32, 64 and 128 columns, not %d", columns)
}

// matmulSPIRV is the binary built for that many columns.
func matmulSPIRV(columns int) ([]byte, error) {
	switch columns {
	case wideColumns:
		return matmulWideSPIRV, nil
	case smallColumns:
		return matmulSmallSPIRV, nil
	}
	return nil, fmt.Errorf("vk: the tiled product is built at %d and %d columns, not %d",
		smallColumns, wideColumns, columns)
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
	}); err != nil {
		return err
	}
	answer := m.out.Floats()
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
	groups := uint32((m.rows+m.perGroup-1)/m.perGroup) * uint32(m.split)
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
	for _, b := range []**Buffer{&m.parts, &m.out, &m.as, &m.aq, &m.stageS, &m.stageQ, &m.weights} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}
