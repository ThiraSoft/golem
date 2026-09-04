package vk

// One quantized matrix on the card, answering several columns a pass.
//
// vk/quanthead.go put a matrix in device memory and read one column of it;
// vk/matmul.go reads many, but only in the tiled forms a prompt takes — Q4_0
// and the K-quants, at thirty-two columns and up. Between those sits the case
// this file is for: a handful of columns, in whatever format the file wrote,
// against weights that are read once for all of them.
//
// That is what several live streams of the same model are. A transcription is
// one column a frame and the trunk is fifty-four megabytes a layer, so a second
// microphone on the processor reads those megabytes a second time and gets
// nothing for it. On the card the weights do not move: the pass is one dispatch
// whatever the width, and the width is the number of streams.
//
// The activation is staged here rather than read from the caller's nn.Batch,
// which interleaves its quantized form by block and then by column. A column of
// that is contiguous only when there is one — the restriction QuantHead and
// Mixture.Run each carry — so a column is copied into the shape the kernel
// reads, which is each column whole, one after another.

import (
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// A QuantBatch is a matrix resident in device memory and the buffers one pass
// over it needs, built for a fixed maximum width.
type QuantBatch struct {
	d          *Device
	q          nn.Quant
	rows, cols int
	width      int // the widest pass this accepts
	padded     int // the binary that pass is run at, which is width rounded up

	weights *Buffer // device local, uploaded once
	aq      *Buffer // the activations' eight-bit magnitudes, column after column
	as      *Buffer // their scales, then their corrections
	out     *Buffer // rows floats a column

	pipe   *Pipeline
	set    *Set
	groups uint32
}

// NewQuantBatch uploads a matrix and binds the mat-vec of its format to it, for
// passes of up to width columns. data is the tensor as the file holds it and q
// the format the file declared — never one inferred from the length, because
// the lengths collide.
func NewQuantBatch(d *Device, q nn.Quant, data []byte, rows, cols, width int) (*QuantBatch, error) {
	if !QuantReadable(q) {
		return nil, fmt.Errorf("vk: there is no product kernel for %s", q)
	}
	if width < 1 {
		return nil, fmt.Errorf("vk: a pass is at least one column, given %d", width)
	}
	layout, err := quantLayout(q, data, rows, cols)
	if err != nil {
		return nil, err
	}
	b := &QuantBatch{d: d, q: q, rows: rows, cols: cols, width: width}
	fail := func(err error) (*QuantBatch, error) {
		b.Close()
		return nil, err
	}
	if b.weights, err = d.Upload(layout); err != nil {
		return nil, err
	}
	layout = nil

	// The buffers are sized for the binary the pass is run at, not for the pass:
	// a kernel built at four columns reads four whether or not four were asked
	// for, and a buffer that stopped at three would have it reading past its
	// end. roundedWidth is asked before anything is allocated for that reason.
	pipe, err := newQuantProduct(d, q, false)
	if err != nil {
		return fail(err)
	}
	b.pipe = pipe
	padded, err := roundedWidth(pipe, width)
	if err != nil {
		return fail(err)
	}
	b.padded = padded

	nb := cols / nn.QuantBlock
	for _, spec := range []struct {
		into **Buffer
		size int
		back bool
	}{
		{&b.aq, cols * padded, false},
		{&b.as, 2 * nb * padded * 4, false},
		{&b.out, rows * padded * 4, true},
	} {
		var buf *Buffer
		if spec.back {
			buf, err = d.Readback(spec.size, bufferUsageStorage)
		} else {
			buf, err = d.Host(spec.size, bufferUsageStorage)
		}
		if err != nil {
			return fail(err)
		}
		*spec.into = buf
	}

	if b.set, err = b.pipe.NewSet([]*Buffer{b.weights, b.aq, b.as, b.out}); err != nil {
		return fail(err)
	}
	outs := quantOuts(q)
	groups := (rows + outs - 1) / outs
	if groups > 65535 {
		return fail(fmt.Errorf("vk: %d rows want %d workgroups, past the 65535 an axis holds", rows, groups))
	}
	b.groups = uint32(groups)
	return b, nil
}

// Width is the widest pass this matrix was built for.
func (b *QuantBatch) Width() int { return b.width }

// Prepare puts the activations in the eight-bit form every kernel behind the
// door reads.
func (b *QuantBatch) Prepare(src *nn.Batch) { src.Quantize() }

// SetColumn stages one column of the batch, which must already carry its Q8_0
// form. Columns are written whole and one after another, which is not how
// nn.Batch holds them.
func (b *QuantBatch) SetColumn(column int, src *nn.Batch) error {
	if src.Q == nil {
		return fmt.Errorf("vk: a %s product needs the activation in its Q8_0 form", b.q)
	}
	if src.Width != b.cols {
		return fmt.Errorf("vk: the product reads %d inputs, the activation has %d", b.cols, src.Width)
	}
	if column < 0 || column >= b.padded {
		return fmt.Errorf("vk: column %d of a pass built for %d", column, b.width)
	}
	if column >= src.Size {
		return fmt.Errorf("vk: column %d of a batch of %d", column, src.Size)
	}
	nb := b.cols / nn.QuantBlock
	dst := b.aq.Bytes()[column*b.cols:]
	scales := b.as.Floats()[column*2*nb:]
	// nn.Batch keeps block-major and then column: block*Stride + column names
	// the block of this column, so the copy walks blocks rather than bytes.
	for block := 0; block < nb; block++ {
		index := block*src.Stride + column
		copy(dst[block*nn.QuantBlock:], magnitudeBytes(src.Q[index*nn.QuantBlock:(index+1)*nn.QuantBlock]))
		scales[block] = src.Scales[index]
		scales[nb+block] = src.Corr[index]
	}
	return nil
}

// Run computes the product for the columns staged and writes one answer each.
//
// The binaries are built at one, two, four, eight and sixteen columns, so a
// pass of three is run as one of four and the answer of the fourth is dropped.
// That is the cheap side of the trade: the weights are read once whatever the
// width, and a column that is thrown away costs its own arithmetic and none of
// the traffic.
func (b *QuantBatch) Run(outs [][]float32) error {
	if len(outs) < 1 || len(outs) > b.width {
		return fmt.Errorf("vk: a pass of %d columns, built for %d", len(outs), b.width)
	}
	for _, o := range outs {
		if len(o) != b.rows {
			return fmt.Errorf("vk: a column writes %d outputs, given %d", b.rows, len(o))
		}
	}
	columns, err := roundedWidth(b.pipe, len(outs))
	if err != nil {
		return err
	}
	push := moePush{dim: uint32(b.rows), ffn: uint32(b.cols), used: 1}
	if err := b.d.Submit(func(r *Recorder) {
		if columns == 1 {
			r.Dispatch(b.set, b.groups, unsafe.Pointer(&push))
			return
		}
		r.DispatchWide(b.set, columns, b.groups, unsafe.Pointer(&push))
	}); err != nil {
		return err
	}
	answer := b.out.Floats()
	for c := range outs {
		copy(outs[c], answer[c*b.rows:])
	}
	return nil
}

// roundedWidth is the narrowest binary this pipeline has that answers at least
// n columns. One is always there — it is the pipeline itself — and the rest are
// what newQuantProduct compiled.
func roundedWidth(p *Pipeline, n int) (int, error) {
	if n <= 1 {
		return 1, nil
	}
	for _, w := range p.Widths() {
		if w >= n {
			return w, nil
		}
	}
	return 0, fmt.Errorf("vk: no pass answers %d columns; this pipeline has 1 and %v", n, p.Widths())
}

func (b *QuantBatch) Close() {
	if b.set != nil {
		b.set.Close()
		b.set = nil
	}
	if b.pipe != nil {
		b.pipe.Close()
		b.pipe = nil
	}
	for _, buf := range []**Buffer{&b.out, &b.as, &b.aq, &b.weights} {
		if *buf != nil {
			(*buf).Close()
			*buf = nil
		}
	}
}

// magnitudeBytes reads a run of signed eight-bit magnitudes as the unsigned
// bytes the buffer holds. The kernel reads them back as signed, so this is a
// reinterpretation and not a conversion — and it is not vk/mixture.go's
// asBytes, which reads floats.
func magnitudeBytes(q []int8) []byte {
	if len(q) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&q[0])), len(q))
}
