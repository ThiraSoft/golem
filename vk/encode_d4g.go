package vk

// The encoder of the D4G format on the card.
//
// Converting a checkpoint is two halves. The first runs the model to see what
// activations each site meets, and it needs the whole model and its engine.
// The second turns every matrix into codes, and it needs nothing but the
// matrix: a scale block of thirty-two weights searches its own step over some
// forty candidates, each costing eight roundings onto the lattice with a shell
// test, and no block ever looks at another. Eight hundred million of those was
// an hour and eight minutes of an eight-core processor, measured, and it is the
// half that does not care where it runs.
//
// The rows arrive raw and leave as codes. What happens between is
// shaders/prepare_d4g.comp — the same scale and rotation the activations will
// meet, already held to the processor by TestPrepareD4GMatchesCPU — and then
// shaders/encode_d4g.comp. Packing the codes into the file's twelve-bit planes
// stays on the processor: it is a shift and a mask per weight, and it would
// cost more to describe to a card than to do.

import (
	_ "embed"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/encode_d4g.comp -o shaders/encode_d4g.spv

//go:embed shaders/encode_d4g.spv
var encodeD4GSPIRV []byte

type encodeD4GPush struct {
	blocks uint32
	lim    int32
	side   int32
	full   int32
	edge   int32
	beta   float32
	spanLo float32
	spanHi float32
}

// D4GEncoder holds the tables and the room for one pass of rows.
type D4GEncoder struct {
	d    *Device
	bits int
	cap  int // weights one pass may carry

	inv       *Buffer
	stept     *Buffer
	stage     *Buffer // host, what a pass of rows is written into
	x         *Buffer // device, what the kernels read and write
	codes     *Buffer
	steps     *Buffer
	codesBack *Buffer
	stepsBack *Buffer

	pipe *Pipeline
	set  *Set

	lim, side, full, edge int
}

// D4GEncodeBlock is how many weights share one step code, which is what the
// kernel is written for. compress.D4Params.ScaleBlock may say otherwise, and a
// caller that says otherwise does not get this path.
const D4GEncodeBlock = 32

// NewD4GEncoder builds the encoder for a code width. cap is how many weights
// one pass carries: the buffers are five and an eighth bytes a weight, so
// sixty-four million of them is a third of a gigabyte and a matrix of any size
// goes through in passes of that.
func NewD4GEncoder(d *Device, bits, capacity int) (*D4GEncoder, error) {
	if bits != nn.D4Bits && bits != nn.D4Bits16 {
		return nil, fmt.Errorf("vk: %d is not a code width this encoder knows", bits)
	}
	if capacity%D4GEncodeBlock != 0 {
		return nil, fmt.Errorf("vk: a pass of %d weights is not a whole number of scale blocks", capacity)
	}
	e := &D4GEncoder{d: d, bits: bits, cap: capacity}
	lim, side, table := nn.D4InverseTable(bits)
	e.lim, e.side = lim, side
	e.full, e.edge = nn.D4TierNorms(bits)

	var err error
	if e.inv, err = d.Upload(asBytesUint32(table)); err != nil {
		e.Close()
		return nil, err
	}
	if e.stept, err = d.Upload(asBytes(nn.D4StepTable())); err != nil {
		e.Close()
		return nil, err
	}
	// The rows live on the card while they are worked on, and only cross the
	// bus twice. Reading them straight out of host memory instead costs five
	// times the kernel: it is a hundred and seventy megabytes a second, which
	// is what an uncached read across the bus is, and the arithmetic never
	// gets a chance to be the thing that is slow.
	for _, b := range []struct {
		at    **Buffer
		size  int
		usage uint32
	}{
		{&e.stage, capacity * 4, bufferUsageTransferSrc},
		{&e.x, capacity * 4, bufferUsageStorage | bufferUsageTransferDst},
		{&e.codes, capacity / 4 * 2, bufferUsageStorage | bufferUsageTransferSrc},
		{&e.steps, capacity / D4GEncodeBlock * 4, bufferUsageStorage | bufferUsageTransferSrc},
		{&e.codesBack, capacity / 4 * 2, bufferUsageTransferDst},
		{&e.stepsBack, capacity / D4GEncodeBlock * 4, bufferUsageTransferDst},
	} {
		var err error
		if b.usage&bufferUsageStorage != 0 {
			*b.at, err = d.Local(b.size, b.usage)
		} else {
			*b.at, err = d.Host(b.size, b.usage)
		}
		if err != nil {
			e.Close()
			return nil, err
		}
	}
	bufs := []*Buffer{e.x, e.inv, e.stept, e.codes, e.steps}
	if e.pipe, err = d.NewPipeline(encodeD4GSPIRV, len(bufs), uint32(unsafe.Sizeof(encodeD4GPush{}))); err != nil {
		e.Close()
		return nil, err
	}
	if e.set, err = e.pipe.NewSet(bufs); err != nil {
		e.Close()
		return nil, err
	}
	return e, nil
}

// Capacity is how many weights one pass carries.
func (e *D4GEncoder) Capacity() int { return e.cap }

// D4GEncodeParams is what the encoder needs that the file does not carry: the
// rotation and the span of the step search. They are compress.D4Params under
// another name, because vk has no reason to import the package that converts.
type D4GEncodeParams struct {
	HadGroup       int     // 0 leaves the matrix unrotated
	Beta           float32 // how far a block is scaled up before rounding
	SpanLo, SpanHi float32 // the step search's span, as a fraction of rms/beta
}

// Encode writes the codes and the step of every scale block of a matrix, into
// caller-provided slices: one code per four weights and one step per
// thirty-two. Packing them into the file's planes is the caller's, because it
// is the caller that knows the format's block.
//
// q is the per-column vector the weights are scaled by — the reciprocal of what
// the activations will meet — or nil for a matrix that is neither scaled nor
// rotated.
func (e *D4GEncoder) Encode(w []float32, rows, cols int, q []float32, p D4GEncodeParams, codes []uint16, steps []byte) error {
	if len(w) != rows*cols {
		return fmt.Errorf("vk: a matrix of %dx%d is not %d weights", rows, cols, len(w))
	}
	if cols%D4GEncodeBlock != 0 {
		return fmt.Errorf("vk: a row of %d is not a whole number of scale blocks", cols)
	}
	if len(codes) != rows*cols/4 || len(steps) != rows*cols/D4GEncodeBlock {
		return fmt.Errorf("vk: room for %d codes and %d steps, wanted %d and %d",
			len(codes), len(steps), rows*cols/4, rows*cols/D4GEncodeBlock)
	}
	if cols > e.cap {
		return fmt.Errorf("vk: a row of %d does not fit a pass of %d", cols, e.cap)
	}
	if q != nil && len(q) != cols {
		return fmt.Errorf("vk: a vector of %d for a row of %d", len(q), cols)
	}

	// A pass is whole rows, because the rotation is a row's own business, and
	// no more rows than the widest dispatch either kernel needs for them. The
	// rotation is the demanding one — a workgroup per group of a hundred and
	// twenty-eight, so a pass of thirty-two million weights would ask for a
	// quarter of a million of them, where Vulkan promises only 65535. This
	// card allows far more and the answer was right; a driver that keeps to
	// the promise would have been silently wrong, which is the one kind of
	// wrong this format cannot afford.
	perPass := e.cap / cols
	if fits := maxWorkgroups * PrepareD4GGroup / cols; fits < perPass {
		perPass = fits
	}
	if perPass < 1 {
		return fmt.Errorf("vk: a row of %d needs more workgroups than a dispatch may have", cols)
	}
	push := encodeD4GPush{
		lim: int32(e.lim), side: int32(e.side),
		full: int32(e.full), edge: int32(e.edge),
		beta: p.Beta, spanLo: p.SpanLo, spanHi: p.SpanHi,
	}
	// The scale and the rotation the activations will meet, so that what the
	// kernel rounds is what the reader will. It is bound to the buffer rather
	// than to a pass, so it is built once a matrix.
	var pre *PrepareD4G
	if q != nil && p.HadGroup > 0 {
		var err error
		if pre, err = NewPrepareD4G(e.d, e.x, q, p.HadGroup); err != nil {
			return err
		}
		defer pre.Close()
	}
	scaled := make([]float32, 0)
	if q != nil && pre == nil {
		// A vector with no rotation behind it, which no kernel does: the
		// scaling happens here on the way into the staging buffer.
		scaled = make([]float32, cols)
	}

	xs := e.stage.Floats()
	// The codes come back as they will be read: two sixteen-bit codes a word,
	// little-endian, which is what the caller's slice already is. So they are
	// one copy rather than a walk — and the walk was the whole cost. Reading a
	// host-visible buffer four bytes at a time is 155 MB/s on this card, which
	// took twenty-five times the kernel's own eleven milliseconds; the same
	// bytes as one memmove take thirty.
	out := unsafe.Slice((*byte)(unsafe.Pointer(&codes[0])), len(codes)*2)
	ss := e.stepsBack.Bytes()
	stepWords := make([]byte, 0)

	for at := 0; at < rows; at += perPass {
		n := perPass
		if at+n > rows {
			n = rows - at
		}
		if len(scaled) == 0 {
			copy(xs[:n*cols], w[at*cols:(at+n)*cols])
		} else {
			for r := 0; r < n; r++ {
				src := w[(at+r)*cols : (at+r+1)*cols]
				for j := range src {
					scaled[j] = src[j] * q[j]
				}
				copy(xs[r*cols:(r+1)*cols], scaled)
			}
		}

		blocks := n * cols / D4GEncodeBlock
		push.blocks = uint32(blocks)
		groups := uint32((blocks + encodeD4GLocal - 1) / encodeD4GLocal)
		var pp prepareD4GPush
		if pre != nil {
			pp = pre.Push(n)
		}
		// One submission: the rows across, both kernels, the codes back. A
		// fence a pass rather than a fence a step.
		err := e.d.Submit(func(r *Recorder) {
			r.CopyFrom(e.x, 0, e.stage, 0, n*cols*4)
			r.Barrier()
			if pre != nil {
				r.Dispatch(pre.Set(), pre.Groups(n), unsafe.Pointer(&pp))
				r.Barrier()
			}
			r.Dispatch(e.set, groups, unsafe.Pointer(&push))
			r.Barrier()
			r.CopyFrom(e.codesBack, 0, e.codes, 0, n*cols/4*2)
			r.CopyFrom(e.stepsBack, 0, e.steps, 0, blocks*4)
		})
		if err != nil {
			return err
		}

		copy(out[at*cols/4*2:], ss2(e.codesBack.Bytes(), n*cols/4*2))
		// The steps are a word each and there are thirty-two times fewer of
		// them, so they are narrowed here rather than in the kernel.
		if cap(stepWords) < blocks*4 {
			stepWords = make([]byte, blocks*4)
		}
		stepWords = stepWords[:blocks*4]
		copy(stepWords, ss[:blocks*4])
		for i := 0; i < blocks; i++ {
			steps[at*cols/D4GEncodeBlock+i] = stepWords[i*4]
		}
	}
	return nil
}

// PrepareD4GGroup is the rotation width the activation kernel was built for,
// and so the only one this encoder can put in front of a matrix.
const PrepareD4GGroup = prepareD4GGroup

// maxWorkgroups is what every Vulkan implementation promises on the first axis
// of a dispatch. Cards allow more and this one allows a great deal more, but a
// pass that stays inside the promise costs nothing to arrange.
const maxWorkgroups = 65535

// encodeD4GLocal is the workgroup the kernel declares.
const encodeD4GLocal = 64

// ss2 is the first n bytes of a host-visible buffer, so that the copy above
// reads it as one run rather than as a slice expression at every call.
func ss2(b []byte, n int) []byte { return b[:n] }

func (e *D4GEncoder) Close() {
	for _, c := range []interface{ Close() }{e.set, e.pipe, e.stepsBack, e.codesBack, e.steps, e.codes, e.x, e.stage, e.stept, e.inv} {
		switch v := c.(type) {
		case *Set:
			if v != nil {
				v.Close()
			}
		case *Pipeline:
			if v != nil {
				v.Close()
			}
		case *Buffer:
			if v != nil {
				v.Close()
			}
		}
	}
}
