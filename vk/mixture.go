package vk

// The feed-forward half of a mixture block, on the card.
//
// This is where the bytes are. A token of the 26B A4B reads about 2.3
// gigabytes; the logit head is 0.6 of them, the experts 0.8, and the shared
// branch beside them 0.3. The experts are read eight matrices at a time out of
// a hundred and twenty-eight, thirty times over, and on the CPU it is all
// bandwidth — the same 37 gigabytes a second the head was stuck at.
//
// Both branches are here because they are one submission. They leave the same
// residual under two different norms and their outputs are added, so nothing
// in either waits on the other: the card can run them together, and a
// submission costs sixty-three microseconds whatever is in it. Two of them a
// block would be four milliseconds a token spent asking rather than computing.
//
// The whole stack is resident — 11.96 gibibytes of experts on this checkpoint
// and 0.3 more of shared branches, which is why a card with sixteen is the
// smallest one that can do this. The routing stays on the CPU: it reads the
// residual, which the CPU already has, and choosing eight of a hundred and
// twenty-eight is a hundred and twenty-eight comparisons.

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_gateup.comp -o shaders/moe_gateup.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_down.comp -o shaders/moe_down.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec.spv

//go:embed shaders/moe_gateup.spv
var moeGateUpSPIRV []byte

//go:embed shaders/moe_down.spv
var moeDownSPIRV []byte

// matvecSPIRV is the plain Q4_0 product, which the shared branch's down
// projection and the attention's four projections both want.
//
//go:embed shaders/matvec.spv
var matvecSPIRV []byte

// The same two kernels built for a batch of columns. shaders/matvec.comp says
// why the count is compiled in rather than pushed.
//
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec8.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/moe_gateup.comp -o shaders/moe_gateup8.spv

//go:embed shaders/matvec8.spv
var matvecWideSPIRV []byte

//go:embed shaders/moe_gateup8.spv
var moeGateUpWideSPIRV []byte

// The tiled product, which is the shape a prompt wants where the mat-vec is
// the shape a token wants. shaders/matmul.comp says why.
//
//go:embed shaders/matmul32.spv
var matmulWideSPIRV []byte

//go:embed shaders/matmul8.spv
var matmulSmallSPIRV []byte

// expertsUsed is what the mixture's down kernel is written for: its workgroup
// is eight outputs by eight experts. A checkpoint that chose a different
// number would need the shape changed, not a constant.
const expertsUsed = 8

// downOuts is how many outputs shaders/moe_down.comp writes per workgroup.
const downOuts = 8

// matvecOuts is shaders/matvec.comp's, which serves the shared branch's down
// projection and every projection of an attention.
const matvecOuts = 16

// A Mixture is every feed-forward matrix of a model, resident, plus the
// kernels that read them and the small buffers a token passes through.
type Mixture struct {
	tl *Timeline // set by Profile, nil everywhere else
	d *Device

	dim, ffn, dense, experts int
	act                      Activation

	gateUp, down, denseDown *Pipeline
	gelu                    *Buffer // ggml's GELU table, uploaded once
	zero                    *Buffer // one identifier, always zero, for the shared branch

	// The expert branch's traffic, reused by every block.
	xq, xs, ids, cw, out *Buffer
	aq, as               *Buffer // the intermediate, which never leaves the card

	// The shared branch's, which is the same shape with one expert.
	dxq, dxs, dout *Buffer
	daq, das       *Buffer

	blocks []*mixtureBlock
}

// A mixtureBlock is one block's matrices and the bindings that read them.
type mixtureBlock struct {
	gateUp, down           *Buffer
	denseGateUp, denseDown *Buffer
	setGateUp, setDown     *Set
	setDenseUp, setDenseDn *Set
}

// moePush is what all three kernels take. matvec.comp reads the first three
// fields and ignores the activation, which is spent before it runs.
type moePush struct {
	dim  uint32
	ffn  uint32
	used uint32
	act  uint32
	// col is the first column of the pass a dispatch answers, which only the
	// gate kernel reads: it is the one kernel that cannot be built at the full
	// width of a pass, so a wide pass runs it more than once at an offset.
	// Every other kernel declares four uints and ignores this one.
	col uint32
}

// An Activation is what a gated feed forward puts on its gate. Gemma 4 looks
// ggml's GELU up in a table; Qwen3 evaluates a SiLU. There is no third.
type Activation uint32

const (
	GELU Activation = iota
	SiLU
)

// NewMixture builds the kernels and the shared buffers. AddBlock then uploads
// one block at a time, so that a caller can report progress over twelve
// gibibytes rather than disappear into them.
//
// ffn is one expert's width and dense the shared branch's, which are not the
// same number: 704 against 2112 on this checkpoint.
func NewMixture(d *Device, dim, ffn, dense, experts, used int, act Activation) (*Mixture, error) {
	if experts > 0 && used != expertsUsed {
		return nil, fmt.Errorf("vk: the expert kernels are written for %d experts a token, this model uses %d", expertsUsed, used)
	}
	shapes := []int{dim, dense}
	if experts > 0 {
		shapes = append(shapes, ffn)
	}
	for _, n := range shapes {
		if n%nn.QuantBlock != 0 {
			return nil, fmt.Errorf("vk: feed-forward shapes must be multiples of %d, given %d", nn.QuantBlock, n)
		}
	}
	if dim%downOuts != 0 {
		return nil, fmt.Errorf("vk: the down kernel writes %d outputs at a time, and %d is not a multiple of it", downOuts, dim)
	}
	m := &Mixture{d: d, dim: dim, ffn: ffn, dense: dense, experts: experts, act: act}

	push := uint32(unsafe.Sizeof(moePush{}))
	var err error
	pipes := []struct {
		into     **Pipeline
		spirv    []byte
		bindings int
	}{
		{&m.gateUp, moeGateUpSPIRV, 7},
		{&m.denseDown, matvecSPIRV, 4},
	}
	if experts > 0 {
		pipes = append(pipes, struct {
			into     **Pipeline
			spirv    []byte
			bindings int
		}{&m.down, moeDownSPIRV, 6})
	}
	for _, spec := range pipes {
		if *spec.into, err = d.NewPipeline(spec.spirv, spec.bindings, push); err != nil {
			m.Close()
			return nil, err
		}
	}

	// The shared branch reads its weights once for a whole batch of positions
	// when there is one. An expert branch never does: each position routes to
	// its own eight matrices, so there is nothing between two of them to share.
	//
	// The down projection has two wide forms and the gate has one: above eight
	// columns the down projection is the tiled product of shaders/matmul.comp,
	// which takes the same bindings and the same push block as the mat-vec, so
	// one Set reaches both. The gate is its own kernel — it fuses two matrices
	// and an activation — and stops at eight, because its shared memory grows
	// with the width and eight already spends sixteen kilobytes of the
	// thirty-two a workgroup has. A wider pass runs it repeatedly at an offset.
	for _, spec := range []struct {
		pipe    **Pipeline
		columns int
		spirv   []byte
	}{
		{&m.gateUp, smallColumns, moeGateUpWideSPIRV},
		{&m.denseDown, smallColumns, matvecWideSPIRV},
		{&m.denseDown, wideColumns, matmulWideSPIRV},
	} {
		if err := (*spec.pipe).Wide(spec.columns, spec.spirv); err != nil {
			m.Close()
			return nil, err
		}
	}

	if m.gelu, err = d.Upload(asBytes(nn.GELUTableData())); err != nil {
		m.Close()
		return nil, err
	}
	if m.zero, err = d.Upload(make([]byte, 4)); err != nil {
		m.Close()
		return nil, err
	}

	in := dim / nn.QuantBlock
	dmid := dense / nn.QuantBlock
	type bufSpec struct {
		into  **Buffer
		size  int
		local bool
	}
	bufs := []bufSpec{
		{&m.dxq, dim * maxColumns, false},         // the shared branch's input
		{&m.dxs, 2 * in * 4 * maxColumns, false},  //
		{&m.dout, dim * 4 * maxColumns, false},    // its output
		{&m.daq, dense * maxColumns, true},        // its intermediate
		{&m.das, 2 * dmid * 4 * maxColumns, true}, //
	}
	if experts > 0 {
		mid := ffn / nn.QuantBlock
		bufs = append(bufs,
			bufSpec{&m.xq, dim, false},                      // the expert branch's input
			bufSpec{&m.xs, 2 * in * 4, false},               // its scales, then its corrections
			bufSpec{&m.ids, expertsUsed * 4, false},         // the chosen experts
			bufSpec{&m.cw, expertsUsed * 4, false},          // routing weight times expert scale
			bufSpec{&m.out, dim * 4, false},                 // that branch's output
			bufSpec{&m.aq, expertsUsed * ffn, true},         // its intermediate
			bufSpec{&m.as, 2 * expertsUsed * mid * 4, true}, // and that intermediate's scales
		)
	}
	for _, spec := range bufs {
		var b *Buffer
		if spec.local {
			b, err = d.Local(spec.size, bufferUsageStorage)
		} else {
			b, err = d.Host(spec.size, bufferUsageStorage)
		}
		if err != nil {
			m.Close()
			return nil, err
		}
		*spec.into = b
	}
	return m, nil
}

// Blocks is how many have been added.
func (m *Mixture) Blocks() int { return len(m.blocks) }

// AddBlock uploads one block's matrices, in the file's own layout, and binds
// the kernels to them. gate and up are the shared branch's two halves, which
// are concatenated here into the one matrix of twice the width the kernel
// reads. The blocks are read back in the order they were added.
// A dense checkpoint has no expert stacks: gateUpExps and downExps are nil,
// and the block is the shared branch alone.
func (m *Mixture) AddBlock(gateUpExps, downExps, gate, up, down []byte) error {
	shapes := []struct {
		what       string
		data       []byte
		rows, cols int
	}{
		{"the shared gate", gate, m.dense, m.dim},
		{"the shared up", up, m.dense, m.dim},
		{"the shared down", down, m.dim, m.dense},
	}
	if m.experts > 0 {
		shapes = append(shapes,
			struct {
				what       string
				data       []byte
				rows, cols int
			}{"the gate-and-up stack", gateUpExps, m.experts * 2 * m.ffn, m.dim},
			struct {
				what       string
				data       []byte
				rows, cols int
			}{"the down stack", downExps, m.experts * m.dim, m.ffn},
		)
	} else if gateUpExps != nil || downExps != nil {
		return fmt.Errorf("vk: this mixture was opened without experts, and the block brings %d bytes of them", len(gateUpExps)+len(downExps))
	}
	for _, spec := range shapes {
		if want := spec.rows * rowBytesQ4_0(spec.cols); len(spec.data) != want {
			return fmt.Errorf("vk: %s should be %d bytes, given %d", spec.what, want, len(spec.data))
		}
	}

	b := &mixtureBlock{}
	fail := func(err error) error {
		b.close()
		return err
	}
	var err error
	if m.experts > 0 {
		if b.gateUp, err = m.d.Upload(splitQ4_0(gateUpExps, m.experts*2*m.ffn, m.dim)); err != nil {
			return fail(err)
		}
		if b.down, err = m.d.Upload(splitQ4_0(downExps, m.experts*m.dim, m.ffn)); err != nil {
			return fail(err)
		}
	}
	// The shared branch's gate and up, one after the other, which is the
	// layout the mixture's own stack already has.
	joined := make([]byte, 0, len(gate)+len(up))
	joined = append(append(joined, gate...), up...)
	if b.denseGateUp, err = m.d.Upload(splitQ4_0(joined, 2*m.dense, m.dim)); err != nil {
		return fail(err)
	}
	if b.denseDown, err = m.d.Upload(splitQ4_0(down, m.dim, m.dense)); err != nil {
		return fail(err)
	}

	type setSpec struct {
		into *(*Set)
		pipe *Pipeline
		bufs []*Buffer
	}
	sets := []setSpec{
		{&b.setDenseUp, m.gateUp, []*Buffer{b.denseGateUp, m.dxq, m.dxs, m.zero, m.gelu, m.daq, m.das}},
		{&b.setDenseDn, m.denseDown, []*Buffer{b.denseDown, m.daq, m.das, m.dout}},
	}
	if m.experts > 0 {
		sets = append(sets,
			setSpec{&b.setGateUp, m.gateUp, []*Buffer{b.gateUp, m.xq, m.xs, m.ids, m.gelu, m.aq, m.as}},
			setSpec{&b.setDown, m.down, []*Buffer{b.down, m.aq, m.as, m.ids, m.cw, m.out}},
		)
	}
	for _, spec := range sets {
		if *spec.into, err = spec.pipe.NewSet(spec.bufs); err != nil {
			return fail(err)
		}
	}
	m.blocks = append(m.blocks, b)
	return nil
}

// Run computes one block's whole feed-forward half for one position.
//
// shared and expert carry the two branches' normed inputs in their Q8_0 form,
// one column each; they are different vectors because the two branches read
// the residual under different norms. ids are the chosen experts and weights
// their routing weights with the per-expert scale already folded in. The two
// outputs are written, unnormed and unadded: the norms between here and the
// stream are the block's business.
func (m *Mixture) Run(block int, shared, expert *nn.Batch, ids []int32, weights []float32, sharedOut, expertOut []float32) error {
	if m.experts == 0 {
		return fmt.Errorf("vk: this mixture has no experts, and Run computes both branches")
	}
	if block < 0 || block >= len(m.blocks) {
		return fmt.Errorf("vk: block %d of %d", block, len(m.blocks))
	}
	for _, b := range []*nn.Batch{shared, expert} {
		if b.Width != m.dim || b.Size != 1 {
			return fmt.Errorf("vk: a branch reads one column of %d, given %d of %d", m.dim, b.Size, b.Width)
		}
		if b.Q == nil {
			return fmt.Errorf("vk: a branch needs its input in the Q8_0 form")
		}
	}
	if len(ids) != expertsUsed || len(weights) != expertsUsed {
		return fmt.Errorf("vk: %d experts expected, given %d ids and %d weights", expertsUsed, len(ids), len(weights))
	}
	for _, o := range [][]float32{sharedOut, expertOut} {
		if len(o) != m.dim {
			return fmt.Errorf("vk: a branch writes %d outputs, given %d", m.dim, len(o))
		}
	}

	in := m.dim / nn.QuantBlock
	for _, spec := range []struct {
		from *nn.Batch
		q    *Buffer
		s    *Buffer
	}{{expert, m.xq, m.xs}, {shared, m.dxq, m.dxs}} {
		q := spec.q.Bytes()
		for i, v := range spec.from.Q[:m.dim] {
			q[i] = byte(v)
		}
		scales := spec.s.Floats()
		copy(scales[:in], spec.from.Scales[:in])
		copy(scales[in:2*in], spec.from.Corr[:in])
	}

	chosen := unsafe.Slice((*int32)(unsafe.Pointer(&m.ids.Bytes()[0])), expertsUsed)
	copy(chosen, ids)
	copy(m.cw.Floats()[:expertsUsed], weights)

	experts := moePush{dim: uint32(m.dim), ffn: uint32(m.ffn), used: expertsUsed, act: uint32(m.act)}
	sharedPush := moePush{dim: uint32(m.dim), ffn: uint32(m.dense), used: 1, act: uint32(m.act)}
	b := m.blocks[block]
	err := m.d.Submit(func(r *Recorder) {
		// Nothing in either branch waits on the other, so they go in without a
		// barrier between them and the card runs them together.
		r.Dispatch(b.setGateUp, uint32(expertsUsed*m.ffn/nn.QuantBlock), unsafe.Pointer(&experts))
		r.Dispatch(b.setDenseUp, uint32(m.dense/nn.QuantBlock), unsafe.Pointer(&sharedPush))
		r.Barrier()
		r.Dispatch(b.setDown, uint32(m.dim/downOuts), unsafe.Pointer(&experts))
		r.Dispatch(b.setDenseDn, uint32((m.dim+matvecOuts-1)/matvecOuts), unsafe.Pointer(&sharedPush))
	})
	if err != nil {
		return err
	}
	copy(expertOut, m.out.Floats()[:m.dim])
	copy(sharedOut, m.dout.Floats()[:m.dim])
	return nil
}

// RunTimes repeats one block's submission n times inside a single one, with
// the inputs already in place from a previous Run.
//
// It measures the kernels rather than the token. A block's work is three
// hundred microseconds and the CPU takes the next thirty back, so the card
// never leaves its low clocks: measured one submission at a time this runs at
// a quarter of the bandwidth it reaches when it is not allowed to rest. Both
// numbers are worth having and they are not the same number — vk/compute.go
// says the same thing about the logit head.
func (m *Mixture) RunTimes(block, n int) error {
	if m.experts == 0 {
		return fmt.Errorf("vk: this mixture has no experts, and RunTimes repeats both branches")
	}
	if block < 0 || block >= len(m.blocks) {
		return fmt.Errorf("vk: block %d of %d", block, len(m.blocks))
	}
	experts := moePush{dim: uint32(m.dim), ffn: uint32(m.ffn), used: expertsUsed, act: uint32(m.act)}
	shared := moePush{dim: uint32(m.dim), ffn: uint32(m.dense), used: 1, act: uint32(m.act)}
	b := m.blocks[block]
	return m.d.Submit(func(r *Recorder) {
		for i := 0; i < n; i++ {
			if i > 0 {
				r.Barrier()
			}
			r.Dispatch(b.setGateUp, uint32(expertsUsed*m.ffn/nn.QuantBlock), unsafe.Pointer(&experts))
			r.Dispatch(b.setDenseUp, uint32(m.dense/nn.QuantBlock), unsafe.Pointer(&shared))
			r.Barrier()
			r.Dispatch(b.setDown, uint32(m.dim/downOuts), unsafe.Pointer(&experts))
			r.Dispatch(b.setDenseDn, uint32((m.dim+matvecOuts-1)/matvecOuts), unsafe.Pointer(&shared))
		}
	})
}

// Record puts one block's two branches into a recording without submitting
// it, which is what running a whole token in one submission needs. The inputs
// and the routing must already be in the buffers the accessors below name.
// Profile is Stack.Profile, forwarded: one stamp between the two halves.
func (m *Mixture) Profile(t *Timeline) { m.tl = t }

// Record puts one block's feed-forward half into a recording, for the given
// number of columns. More than one is a stretch of a prompt, and only the
// shared branch can take it — see the note beside the wide pipelines above.
func (m *Mixture) Record(r *Recorder, block, columns int) {
	experts := moePush{dim: uint32(m.dim), ffn: uint32(m.ffn), used: expertsUsed, act: uint32(m.act)}
	shared := moePush{dim: uint32(m.dim), ffn: uint32(m.dense), used: 1, act: uint32(m.act)}
	b := m.blocks[block]
	width := passWidth(columns)
	up, down := uint32(m.dense/nn.QuantBlock), productGroups(width, m.dim)
	// Nothing in either branch waits on the other, so they go in without a
	// barrier between them and the card runs them together.
	if m.experts > 0 {
		r.Dispatch(b.setGateUp, uint32(expertsUsed*m.ffn/nn.QuantBlock), unsafe.Pointer(&experts))
	}
	if width > 1 {
		// The gate at eight columns at a time, however wide the pass is.
		for c := 0; c < width; c += smallColumns {
			at := shared
			at.col = uint32(c)
			r.DispatchWide(b.setDenseUp, smallColumns, up, unsafe.Pointer(&at))
		}
	} else {
		r.Dispatch(b.setDenseUp, up, unsafe.Pointer(&shared))
	}
	r.Barrier()
	m.tl.Stamp(r, "moe gate/up")
	if m.experts > 0 {
		r.Dispatch(b.setDown, uint32(m.dim/downOuts), unsafe.Pointer(&experts))
	}
	if width > 1 {
		r.DispatchWide(b.setDenseDn, width, down, unsafe.Pointer(&shared))
	} else {
		r.Dispatch(b.setDenseDn, down, unsafe.Pointer(&shared))
	}
}

// The buffers a kernel upstream writes and one downstream reads, so that a
// whole token can be recorded without anything crossing the bus.

// ExpertInput is where the expert branch reads its normed input, Q8_0.
func (m *Mixture) ExpertInput() (*Buffer, *Buffer) { return m.xq, m.xs }

// SharedInput is the same for the branch beside it, which reads the residual
// under a different norm.
func (m *Mixture) SharedInput() (*Buffer, *Buffer) { return m.dxq, m.dxs }

// Routing is the chosen experts and their weights, which the router writes.
func (m *Mixture) Routing() (*Buffer, *Buffer) { return m.ids, m.cw }

// Outputs are the two branches' answers, unnormed and unadded.
func (m *Mixture) Outputs() (shared, experts *Buffer) { return m.dout, m.out }

func (m *Mixture) Close() {
	for _, b := range m.blocks {
		b.close()
	}
	m.blocks = nil
	for _, b := range []**Buffer{
		&m.das, &m.daq, &m.dout, &m.dxs, &m.dxq,
		&m.as, &m.aq, &m.out, &m.cw, &m.ids, &m.xs, &m.xq,
		&m.zero, &m.gelu,
	} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	for _, p := range []**Pipeline{&m.denseDown, &m.down, &m.gateUp} {
		if *p != nil {
			(*p).Close()
			*p = nil
		}
	}
}

func (b *mixtureBlock) close() {
	for _, s := range []**Set{&b.setDenseDn, &b.setDenseUp, &b.setDown, &b.setGateUp} {
		if *s != nil {
			(*s).Close()
			*s = nil
		}
	}
	for _, x := range []**Buffer{&b.denseDown, &b.denseGateUp, &b.down, &b.gateUp} {
		if *x != nil {
			(*x).Close()
			*x = nil
		}
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
// block count is even on every shape this reads, so the nibbles start aligned
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
