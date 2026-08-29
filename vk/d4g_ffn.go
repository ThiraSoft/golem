package vk

// A D4G feed forward on the card: norm out, block output in, nothing between
// them crossing the bus.
//
// This is its own component rather than a second path through vk/mixture.go,
// and the reason is the activation. Everything else in the engine hands a
// matrix its input in Q8_0 — values and scales, two buffers — because that is
// what a Q4_0 kernel reads cheaply. A D4G kernel reads floats, and it has to:
// the scale and the rotation the format undoes on this side are float
// arithmetic, and quantizing between them would throw away the precision they
// exist to preserve. So the whole path is floats, and a component that carries
// floats end to end is a shorter thing to write and to read than a flag
// threaded through one that carries something else.
//
// The gate and the up projection are one matrix. They read the same activation
// through the same vector — one site, one scale, one rotation — so stacking
// their rows is exact, and it turns two dispatches into one.

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/swiglu_d4g.comp -o shaders/swiglu_d4g.spv

//go:embed shaders/swiglu_d4g.spv
var swigluD4GSPIRV []byte

type swigluD4GPush struct {
	ffn     uint32
	columns uint32
}

// D4GFFN is one block's feed forward.
type D4GFFN struct {
	d        *Device
	k        *D4GKernels
	dim, ffn int
	columns  int

	// in is the block's input, normed and then prepared in place. out is what
	// the feed forward answers. Both are the caller's.
	in, out *Buffer

	stacked *Buffer // gate and up, 2*ffn a column
	act     *Buffer // the gated intermediate, ffn a column, prepared in place

	prepIn  *PrepareD4G
	prepAct *PrepareD4G

	gateUp *D4GMatrix
	down   *D4GMatrix

	swiglu *Pipeline
	setAct *Set
}

// NewD4GFFN uploads one block's three matrices and binds them. gate, up and
// down are the file's own bytes; preGateUp and preDown are the two vectors the
// checkpoint carries for this block's two sites. in and out belong to the
// caller and are the stream this feed forward reads and writes; columns is the
// widest pass it will be asked for.
func NewD4GFFN(k *D4GKernels, dim, ffn, columns int, gate, up, down []byte,
	preGateUp, preDown []float32, in, out *Buffer) (*D4GFFN, error) {
	if len(preGateUp) != dim || len(preDown) != ffn {
		return nil, fmt.Errorf("vk: the feed forward's vectors are %d and %d wide, want %d and %d",
			len(preGateUp), len(preDown), dim, ffn)
	}
	f := &D4GFFN{d: k.d, k: k, dim: dim, ffn: ffn, columns: columns, in: in, out: out}
	fail := func(err error) (*D4GFFN, error) {
		f.Close()
		return nil, err
	}
	var err error
	if f.stacked, err = k.d.Local(2*ffn*columns*4, bufferUsageStorage); err != nil {
		return fail(err)
	}
	if f.act, err = k.d.Local(ffn*columns*4, bufferUsageStorage); err != nil {
		return fail(err)
	}
	if f.prepIn, err = NewPrepareD4G(k.d, in, preGateUp, prepareD4GGroup); err != nil {
		return fail(err)
	}
	if f.prepAct, err = NewPrepareD4G(k.d, f.act, preDown, prepareD4GGroup); err != nil {
		return fail(err)
	}

	// Gate then up, stacked into one matrix of twice the rows.
	joined := make([]byte, 0, len(gate)+len(up))
	joined = append(append(joined, gate...), up...)
	if f.gateUp, err = NewD4GMatrixOn(k, joined, 2*ffn, dim, in, f.stacked); err != nil {
		return fail(err)
	}
	if f.down, err = NewD4GMatrixOn(k, down, dim, ffn, f.act, out); err != nil {
		return fail(err)
	}

	if f.swiglu, err = k.d.NewPipeline(swigluD4GSPIRV, 2, uint32(unsafe.Sizeof(swigluD4GPush{}))); err != nil {
		return fail(err)
	}
	if f.setAct, err = f.swiglu.NewSet([]*Buffer{f.stacked, f.act}); err != nil {
		return fail(err)
	}
	return f, nil
}

// Record writes the feed forward into a command buffer: prepare the input,
// gate and up in one product, the gate itself, prepare the intermediate, and
// down. The caller has already normed the input into `in`, and reads the
// answer from `out`.
func (f *D4GFFN) Record(r *Recorder, columns int) error {
	width, ok := d4gPassWidth(columns)
	if !ok {
		return fmt.Errorf("vk: %d columns is not a pass this kernel was built for", columns)
	}
	push := f.prepIn.Push(columns)
	r.Dispatch(f.prepIn.Set(), f.prepIn.Groups(columns), unsafe.Pointer(&push))
	r.Barrier()

	for first := 0; first < columns; first += width {
		p := f.gateUp.Push(first)
		r.Dispatch(f.gateUp.Set(width), f.gateUp.Groups(), unsafe.Pointer(&p))
	}
	r.Barrier()

	act := swigluD4GPush{ffn: uint32(f.ffn), columns: uint32(columns)}
	r.Dispatch(f.setAct, uint32((f.ffn*columns+255)/256), unsafe.Pointer(&act))
	r.Barrier()

	pd := f.prepAct.Push(columns)
	r.Dispatch(f.prepAct.Set(), f.prepAct.Groups(columns), unsafe.Pointer(&pd))
	r.Barrier()

	for first := 0; first < columns; first += width {
		p := f.down.Push(first)
		r.Dispatch(f.down.Set(width), f.down.Groups(), unsafe.Pointer(&p))
	}
	return nil
}

// d4gPassWidth is the widest pass that divides a run of columns. A pass reads
// the weights once whatever its width, so wider is better until the kernels
// stop being built for it.
func d4gPassWidth(columns int) (int, bool) {
	if columns <= 0 {
		return 0, false
	}
	best := 0
	for _, w := range D4GWidths {
		if columns%w == 0 && w > best {
			best = w
		}
	}
	return best, best > 0
}

func (f *D4GFFN) Close() {
	if f.setAct != nil {
		f.setAct.Close()
	}
	if f.swiglu != nil {
		f.swiglu.Close()
	}
	for _, m := range []*D4GMatrix{f.gateUp, f.down} {
		if m != nil {
			m.Close()
		}
	}
	for _, p := range []*PrepareD4G{f.prepIn, f.prepAct} {
		if p != nil {
			p.Close()
		}
	}
	for _, b := range []*Buffer{f.stacked, f.act} {
		if b != nil {
			b.Close()
		}
	}
	*f = D4GFFN{}
}

// blockBytesD4G is what one row of a matrix costs at this device's code width,
// for a caller checking a checkpoint before it uploads it.
func (k *D4GKernels) RowBytes(cols int) int {
	return cols / nn.D4Block * nn.D4BlockBytes(k.bits)
}
