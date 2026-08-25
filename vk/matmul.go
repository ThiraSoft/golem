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

//go:generate glslc -O -DCOLUMNS=32 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul32.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul.comp -o shaders/matmul8.spv

// Nothing in the engine binds this yet. BenchmarkMatMul is why, and the header
// of shaders/matmul.comp says the rest.

// The widths the tiled product is built at. shaders/matmul.comp gives a thread
// four rows by four columns, so a pass narrower than four columns has nothing
// to tile and the mat-vec kernel is the right shape for it.
const (
	wideColumns  = 32
	smallColumns = 8
)

// matmulRows is shaders/matmul.comp's BM: how many rows one workgroup writes.
const matmulRows = 32

// A MatMul is one Q4_0 matrix resident on a device with the buffers a batch
// passes through.
type MatMul struct {
	d          *Device
	rows, cols int
	columns    int

	weights, aq, as, out *Buffer
	pipe                 *Pipeline
	set                  *Set
}

// NewMatMul uploads a Q4_0 matrix in the file's own layout and binds the tiled
// product to it, for passes of the given width.
func NewMatMul(d *Device, data []byte, rows, cols, columns int) (*MatMul, error) {
	if cols%nn.QuantBlock != 0 {
		return nil, fmt.Errorf("vk: a Q4_0 row needs a multiple of %d columns, given %d", nn.QuantBlock, cols)
	}
	if want := rows * rowBytesQ4_0(cols); len(data) != want {
		return nil, fmt.Errorf("vk: %d rows of %d columns need %d bytes, given %d", rows, cols, want, len(data))
	}
	spirv, err := matmulSPIRV(columns)
	if err != nil {
		return nil, err
	}

	m := &MatMul{d: d, rows: rows, cols: cols, columns: columns}
	if m.weights, err = d.Upload(splitQ4_0(data, rows, cols)); err != nil {
		return nil, err
	}
	nb := cols / nn.QuantBlock
	for _, spec := range []struct {
		into **Buffer
		size int
		back bool
	}{
		{&m.aq, cols * columns, false},
		{&m.as, 2 * nb * 4 * columns, false},
		{&m.out, rows * 4 * columns, true},
	} {
		var b *Buffer
		if spec.back {
			b, err = d.Readback(spec.size, bufferUsageStorage)
		} else {
			b, err = d.Host(spec.size, bufferUsageStorage)
		}
		if err != nil {
			m.Close()
			return nil, err
		}
		*spec.into = b
	}

	buffers := []*Buffer{m.weights, m.aq, m.as, m.out}
	if m.pipe, err = d.NewPipeline(spirv, len(buffers), uint32(unsafe.Sizeof(moePush{}))); err != nil {
		m.Close()
		return nil, err
	}
	if m.set, err = m.pipe.NewSet(buffers); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
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
	dst := m.aq.Bytes()[column*m.cols:]
	for i, v := range b.Q[:m.cols] {
		dst[i] = byte(v)
	}
	scales := m.as.Floats()[column*2*nb:]
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
	push := moePush{dim: uint32(m.rows), ffn: uint32(m.cols), used: 1}
	groups := uint32((m.rows + matmulRows - 1) / matmulRows)
	if err := m.set.Dispatch(groups, unsafe.Pointer(&push)); err != nil {
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
	push := moePush{dim: uint32(m.rows), ffn: uint32(m.cols), used: 1}
	groups := uint32((m.rows + matmulRows - 1) / matmulRows)
	return m.d.Submit(func(r *Recorder) {
		for i := 0; i < n; i++ {
			if i > 0 {
				r.Barrier()
			}
			r.Dispatch(m.set, groups, unsafe.Pointer(&push))
		}
	})
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
	for _, b := range []**Buffer{&m.out, &m.as, &m.aq, &m.weights} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}
