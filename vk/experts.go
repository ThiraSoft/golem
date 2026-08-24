package vk

// The expert branch of a mixture block, on the card.
//
// This is where the bytes are. A token of the 26B A4B reads about 1.7
// gigabytes; the logit head was 0.6 of them and the experts are 0.8, read
// eight matrices at a time out of a hundred and twenty-eight, thirty times
// over. Nothing else in the model comes close, and on the CPU it is all
// bandwidth: the same 37 gigabytes a second the head was stuck at.
//
// The whole stack is resident. Every expert of every block is uploaded once —
// 11.96 gibibytes on this checkpoint, which is why a card with sixteen is the
// smallest one that can do this — and afterwards a block costs two dispatches
// and about twenty kilobytes across the bus. The routing stays on the CPU: it
// reads the residual, which the CPU already has, and choosing eight of a
// hundred and twenty-eight is a hundred and twenty-eight comparisons.
//
// The two dispatches of a block go in one submission with a barrier between
// them, because a submission costs sixty-three microseconds whatever is in it
// and a token has thirty blocks.

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O -fshader-stage=compute shaders/moe_gateup.comp -o shaders/moe_gateup.spv
//go:generate glslc -O -fshader-stage=compute shaders/moe_down.comp -o shaders/moe_down.spv

//go:embed shaders/moe_gateup.spv
var moeGateUpSPIRV []byte

//go:embed shaders/moe_down.spv
var moeDownSPIRV []byte

// expertsUsed is what the shaders are written for. Gemma 4 chooses eight, and
// the down kernel's workgroup is eight outputs by eight experts; a checkpoint
// that chose a different number would need the shape changed, not a constant.
const expertsUsed = 8

// An Experts is every expert matrix of a model, resident, plus the kernels
// that read them and the small buffers a token passes through.
type Experts struct {
	d *Device

	dim, ffn, experts int

	gateUp, down *Pipeline
	gelu         *Buffer // ggml's GELU table, uploaded once

	// One token's worth of traffic, reused by every block.
	xq, xs, ids, cw, out *Buffer
	aq, as               *Buffer // the intermediate, which never leaves the card

	blocks []*expertBlock
}

// An expertBlock is one block's stack of matrices and the two bindings that
// read them.
type expertBlock struct {
	gateUp, down *Buffer
	setGateUp    *Set
	setDown      *Set
}

// moePush is what both shaders take.
type moePush struct {
	dim uint32
	ffn uint32
}

// NewExperts builds the kernels and the shared buffers. AddBlock then uploads
// one block's matrices at a time, so that a caller can report progress over
// twelve gibibytes rather than disappear into them.
func NewExperts(d *Device, dim, ffn, experts, used int) (*Experts, error) {
	if used != expertsUsed {
		return nil, fmt.Errorf("vk: the expert kernels are written for %d experts a token, this model uses %d", expertsUsed, used)
	}
	if dim%nn.QuantBlock != 0 || ffn%nn.QuantBlock != 0 {
		return nil, fmt.Errorf("vk: expert shapes must be multiples of %d, given %d and %d", nn.QuantBlock, dim, ffn)
	}
	if dim%expertsUsed != 0 {
		return nil, fmt.Errorf("vk: the down kernel writes %d outputs at a time, and %d is not a multiple of it", expertsUsed, dim)
	}
	e := &Experts{d: d, dim: dim, ffn: ffn, experts: experts}

	push := uint32(unsafe.Sizeof(moePush{}))
	var err error
	if e.gateUp, err = d.NewPipeline(moeGateUpSPIRV, 7, push); err != nil {
		return nil, err
	}
	if e.down, err = d.NewPipeline(moeDownSPIRV, 6, push); err != nil {
		e.Close()
		return nil, err
	}

	table := nn.GELUTableData()
	if e.gelu, err = d.Upload(asBytes(table)); err != nil {
		e.Close()
		return nil, err
	}

	blocks := dim / nn.QuantBlock
	mid := ffn / nn.QuantBlock
	for _, spec := range []struct {
		into  **Buffer
		size  int
		local bool
	}{
		{&e.xq, dim, false},                      // the input's magnitudes
		{&e.xs, 2 * blocks * 4, false},           // its scales, then its corrections
		{&e.ids, expertsUsed * 4, false},         // the chosen experts
		{&e.cw, expertsUsed * 4, false},          // routing weight times expert scale
		{&e.out, dim * 4, false},                 // the branch's output
		{&e.aq, expertsUsed * ffn, true},         // the intermediate's magnitudes
		{&e.as, 2 * expertsUsed * mid * 4, true}, // and its scales and corrections
	} {
		var b *Buffer
		if spec.local {
			b, err = d.Local(spec.size, bufferUsageStorage)
		} else {
			b, err = d.Host(spec.size, bufferUsageStorage)
		}
		if err != nil {
			e.Close()
			return nil, err
		}
		*spec.into = b
	}
	return e, nil
}

// Blocks is how many have been added.
func (e *Experts) Blocks() int { return len(e.blocks) }

// AddBlock uploads one block's two stacks, in the file's own layout, and binds
// the kernels to them. The blocks are read back in the order they were added.
func (e *Experts) AddBlock(gateUp, down []byte) error {
	if want := e.experts * 2 * e.ffn * rowBytesQ4_0(e.dim); len(gateUp) != want {
		return fmt.Errorf("vk: the gate-and-up stack should be %d bytes, given %d", want, len(gateUp))
	}
	if want := e.experts * e.dim * rowBytesQ4_0(e.ffn); len(down) != want {
		return fmt.Errorf("vk: the down stack should be %d bytes, given %d", want, len(down))
	}

	b := &expertBlock{}
	var err error
	if b.gateUp, err = e.d.Upload(splitQ4_0(gateUp, e.experts*2*e.ffn, e.dim)); err != nil {
		return err
	}
	if b.down, err = e.d.Upload(splitQ4_0(down, e.experts*e.dim, e.ffn)); err != nil {
		b.close()
		return err
	}
	if b.setGateUp, err = e.gateUp.NewSet([]*Buffer{
		b.gateUp, e.xq, e.xs, e.ids, e.gelu, e.aq, e.as,
	}); err != nil {
		b.close()
		return err
	}
	if b.setDown, err = e.down.NewSet([]*Buffer{
		b.down, e.aq, e.as, e.ids, e.cw, e.out,
	}); err != nil {
		b.close()
		return err
	}
	e.blocks = append(e.blocks, b)
	return nil
}

// Run computes one block's expert branch for one position.
//
// in carries the branch's normed input in its Q8_0 form, one column. ids are
// the chosen experts and weights the routing weights already multiplied by the
// per-expert scale. out is written.
func (e *Experts) Run(block int, in *nn.Batch, ids []int32, weights []float32, out []float32) error {
	if block < 0 || block >= len(e.blocks) {
		return fmt.Errorf("vk: block %d of %d", block, len(e.blocks))
	}
	if in.Width != e.dim || in.Size != 1 {
		return fmt.Errorf("vk: the branch reads one column of %d, given %d of %d", e.dim, in.Size, in.Width)
	}
	if in.Q == nil {
		return fmt.Errorf("vk: the expert branch needs the input in its Q8_0 form")
	}
	if len(ids) != expertsUsed || len(weights) != expertsUsed {
		return fmt.Errorf("vk: %d experts expected, given %d ids and %d weights", expertsUsed, len(ids), len(weights))
	}
	if len(out) != e.dim {
		return fmt.Errorf("vk: the branch writes %d outputs, given %d", e.dim, len(out))
	}

	blocks := e.dim / nn.QuantBlock
	q := e.xq.Bytes()
	for i, v := range in.Q[:e.dim] {
		q[i] = byte(v)
	}
	scales := e.xs.Floats()
	copy(scales[:blocks], in.Scales[:blocks])
	copy(scales[blocks:2*blocks], in.Corr[:blocks])

	chosen := unsafe.Slice((*int32)(unsafe.Pointer(&e.ids.Bytes()[0])), expertsUsed)
	copy(chosen, ids)
	copy(e.cw.Floats()[:expertsUsed], weights)

	push := moePush{dim: uint32(e.dim), ffn: uint32(e.ffn)}
	b := e.blocks[block]
	err := e.d.Submit(func(r *Recorder) {
		r.Dispatch(b.setGateUp, uint32(expertsUsed*e.ffn/nn.QuantBlock), unsafe.Pointer(&push))
		r.Barrier()
		r.Dispatch(b.setDown, uint32(e.dim/expertsUsed), unsafe.Pointer(&push))
	})
	if err != nil {
		return err
	}
	copy(out, e.out.Floats()[:e.dim])
	return nil
}

func (e *Experts) Close() {
	for _, b := range e.blocks {
		b.close()
	}
	e.blocks = nil
	for _, b := range []**Buffer{&e.as, &e.aq, &e.out, &e.cw, &e.ids, &e.xs, &e.xq, &e.gelu} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	if e.down != nil {
		e.down.Close()
		e.down = nil
	}
	if e.gateUp != nil {
		e.gateUp.Close()
		e.gateUp = nil
	}
}

func (b *expertBlock) close() {
	if b.setDown != nil {
		b.setDown.Close()
		b.setDown = nil
	}
	if b.setGateUp != nil {
		b.setGateUp.Close()
		b.setGateUp = nil
	}
	if b.down != nil {
		b.down.Close()
		b.down = nil
	}
	if b.gateUp != nil {
		b.gateUp.Close()
		b.gateUp = nil
	}
}

// rowBytesQ4_0 is what one row of that many inputs occupies.
func rowBytesQ4_0(cols int) int { return cols / nn.QuantBlock * 18 }

// splitQ4_0 rewrites rows so that a shader can reach them.
//
// A Q4_0 block is an fp16 scale followed by sixteen bytes of nibbles, and
// eighteen is not a multiple of four: every other block of a row would begin
// off alignment, and a shader indexing a uint array would pay for it on every
// load. So a row is split in two — all of its scales, then all of its nibbles
// — which is the same bytes in the same number, with both halves aligned. The
// row count is even on every shape this reads, so the nibbles start aligned
// too.
func splitQ4_0(src []byte, rows, cols int) []byte {
	nb := cols / nn.QuantBlock
	stride := nb * 18
	dst := make([]byte, len(src))
	for r := 0; r < rows; r++ {
		in := src[r*stride : (r+1)*stride]
		out := dst[r*stride : (r+1)*stride]
		nibbles := out[2*nb:]
		for b := 0; b < nb; b++ {
			block := in[b*18 : (b+1)*18]
			binary.LittleEndian.PutUint16(out[2*b:], binary.LittleEndian.Uint16(block))
			copy(nibbles[b*16:], block[2:])
		}
	}
	return dst
}

// asBytes views a float slice as the bytes behind it, for an upload.
func asBytes(f []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}
