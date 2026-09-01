package vk

// A whole block's attention on the card: the four products, the norms, the
// rotation, the cache, the scores and the mix, in one submission.
//
// The four products alone were worth moving — nineteen megabytes a block
// against a few kilobytes of everything else — but they left the block cut in
// half, with the scores on this side and a submission either side of them. A
// submission costs sixty-three microseconds whatever is in it, and the card
// drops to half its clocks whenever it is handed one and then left alone. So
// the rest follows the products, not for its own arithmetic but so that the
// arithmetic already there stops waiting.
//
// What that costs is the intricacy, which is now written twice: a query norm
// and a key norm, two rotation geometries whose heads are not the same size, a
// value taken from the key before the key was rotated, fifteen blocks at the
// end that compute no keys and read what two earlier ones left behind, and
// three roundings to fp16 that are not optional because llama.cpp holds its
// cache that way. gemma/attention.go is the other copy, and it is the one the
// tests are written against.

import (
	_ "embed"
	"fmt"
	"math"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/attn_prepare.comp -o shaders/attn_prepare.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/attn_scores.comp -o shaders/attn_scores.spv
//go:generate glslc -O -DCOOP --target-env=vulkan1.1 -fshader-stage=compute shaders/attn_scores.comp -o shaders/attn_scores_coop.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/rope_table.comp -o shaders/rope_table.spv

//go:embed shaders/attn_prepare.spv
var attnPrepareSPIRV []byte

//go:embed shaders/attn_scores.spv
var attnScoresSPIRV []byte

// The same kernel with the query-against-key block on the matrix cores. It is
// a second binary rather than a branch because a card without them cannot
// load a module that names them at all.
//
//go:embed shaders/attn_scores_coop.spv
var attnScoresCoopSPIRV []byte

//go:embed shaders/rope_table.spv
var ropeTableSPIRV []byte

// maxColumns is how many positions of a prompt one pass carries: the widest
// COLUMNS the wide shaders are built at, the WHERE_COLUMNS they index the
// position buffer by, and the width every per-column buffer here is allocated
// for. The three have to agree, and nothing checks it but this comment.
//
// passWidth says which binary a pass of a given length actually runs.
const maxColumns = wideColumns

// scoreColumns is how many columns of a pass one workgroup of the scores
// kernel answers at once, and it is the BR of shaders/attn_scores.comp — the
// two have to agree. It used to mean the opposite thing: the kernel gave a
// workgroup to every column and this said how many columns' scores fit in a
// scratch buffer at a time. The scratch is gone with the online softmax, and
// what is left is the tiling that divides the cache traffic.
const scoreColumns = 32

// passWidth is the widest binary that answers a pass of that many columns.
// There are four: the tiled product at a hundred and twenty-eight and at
// thirty-two, the mat-vec at eight, and the mat-vec at one, which is a token.
// A pass shorter than the binary it runs computes the columns above it too,
// out of buffers that are allocated for them and out of positions the
// per-column kernels never dispatch — so the answer is right and the cost is
// the binary's. Hence the ladder rather than one width: a prompt of sixty-four
// read by the widest binary pays for a hundred and twenty-eight, and measured
// that is a third of the rate the thirty-two-column binary gives it.
func passWidth(columns int) int {
	for _, w := range matmulWidths {
		if columns <= w {
			return w
		}
	}
	return wideColumns
}

// A BlockShape is what one block's attention is, as the kernels need to know
// it. It is the block answering for itself rather than a branch on its number.
type BlockShape struct {
	Heads, KVHeads, HeadDim int
	RoPEDims                int
	// Capacity is the ring this block's cache holds: its window, or the whole
	// context on a block that sees everything.
	Capacity int
	// Rotation says which of the model's rotation geometries this block uses.
	// Gemma 4 has two, and a block is asked once rather than branched on.
	Rotation int
	// ValueIsKey says the value is the key before the key was rotated, and
	// OwnsKV that the block computes keys at all. A block that does not reads
	// KVSource's cache, and is given the same buffers rather than a copy.
	ValueIsKey bool
	OwnsKV     bool
	// NormValue says the value projection is RMS-normed, with no gain of its
	// own, before it goes into the cache. Gemma 4 does that; Qwen3 hands the
	// projection straight over, which is what llama.cpp's graph shows.
	NormValue bool
	KVSource  int
	Eps       float32
	// Scale multiplies the scores before the softmax. Gemma 4 scales by one
	// and lets its query norm hold them in range; Qwen3 passes 1/sqrt(head_dim)
	// the way llama.cpp does.
	Scale float32
}

// Columns is how many positions one pass may carry, which callers need in
// order to cut a prompt into batches of it.
func Columns() int { return maxColumns }

// An Attention is every attention matrix and every key-value cache of a model,
// resident, and the buffers one position passes through.
type Attention struct {
	tl *Timeline // set by Profile, nil everywhere else

	where *Buffer // one uvec4 a block: the position and the visible range
	d     *Device

	dim         int // the stream's width, which is what the output projection makes
	maxHeads    int // the widest block's query projection, for the shared buffers
	maxKV       int
	slotContext int // the positions one conversation holds
	slots       int // how many conversations the caches are cut into

	matvec, prepare, scores *Pipeline
	// scoresFloat is the same kernel with a float copy of the mix beside its
	// Q8_0 one, for the blocks whose output projection reads floats. Built
	// only when one is added.
	scoresFloat *Pipeline
	// rope fills the angle tables from the position buffer, at the head of a
	// pass. shaders/rope_table.comp says what it replaced.
	rope *Pipeline
	// reduce folds the slices of a split output projection. See matmulSplit:
	// that projection is the stack's row-poorest tiled product, and the one
	// dispatched with nothing beside it, so it is the one that can be cut.
	reduce   *Pipeline
	splitOut int
	coop     bool

	// One position's traffic. Only the two ends of it cross the bus: the
	// normed stream in, and the output projection's answer back.
	xq, xs, out *Buffer
	// xf and af are the Golem path's: the normed stream as floats, and the mix
	// as floats. Both are nil until a Golem block is added.
	xf, af      *Buffer
	rcos, rsin  []*Buffer // one pair per rotation geometry
	rinv        []*Buffer // its inverse frequencies, written once
	ropeSets    []*Set
	ropeHalf    []uint32 // how many entries of a column each geometry fills
	q, k, v, qh *Buffer
	outParts    *Buffer // the slices of a split output projection
	reduceSet   *Set
	scoreRows   *Buffer
	aq, as      *Buffer

	blocks []*attentionBlock
}

// An attentionBlock is one block's matrices, norms and cache.
type attentionBlock struct {
	shape BlockShape

	q, k, v, o   *Buffer
	qnorm, knorm *Buffer
	ck, cv       *Buffer // nil when the block reads another's cache
	setQ, setK   *Set
	setV, setO   *Set
	setPrepare   *Set
	setScores    *Set
	setOParts    *Set // the output projection when its shared dimension is split

	// golem is the block's Golem side when its projections are in that format,
	// and nil when they are Q4_0. vk/attention_golem.go is all of it.
	golem *golemAttn
}

// maxBlocks is how many entries the position buffer holds, which caps the
// blocks a stack may have.
const maxBlocks = 128

// attnPush is what the two attention kernels take. The two shaders declare the
// same block; the fields past what each reads are ignored.
type attnPush struct {
	heads      uint32
	kvHeads    uint32
	headDim    uint32
	ropeDims   uint32
	block      uint32
	capacity   uint32
	mask       uint32
	valueIsKey uint32
	normValue  uint32
	rotStride  uint32
	eps        float32
}

// ropePush is what the angle table kernel takes: how many entries of a column
// the geometry fills, and how far apart two columns are.
type ropePush struct {
	halfDims uint32
	stride   uint32
}

// scorePush is the second kernel's, which needs a range rather than a position.
type scorePush struct {
	heads    uint32
	kvHeads  uint32
	headDim  uint32
	perKV    uint32
	block    uint32
	capacity uint32
	mask     uint32
	columns  uint32
	scale    float32
	col0     uint32
}

// NewAttention builds the kernels and the shared buffers, which are sized for
// the widest block and the deepest context.
//
// maxQueryHeads is the head count rather than a width: the scores are one
// float per head per position per column, and sizing that buffer off
// maxHeads — which is heads times the head dimension — is a hundred and
// twenty-eight times more memory than it needs at thirty-two columns.
//
// slotContext is what one conversation holds and slots is how many of them
// there are. A block's cache is that many rings laid end to end, and a column
// says which one it belongs to; the context is cut rather than multiplied, so
// four conversations of a thousand positions cost what one of four thousand
// did. gemma/slots.go says why it is cut that way.
func NewAttention(d *Device, dim, maxHeads, maxKV, maxQueryHeads, slotContext, slots, rotations int) (*Attention, error) {
	if slots < 1 {
		return nil, fmt.Errorf("vk: %d slots", slots)
	}
	for _, n := range []int{dim, maxHeads, maxKV} {
		if n%nn.QuantBlock != 0 {
			return nil, fmt.Errorf("vk: attention shapes must be multiples of %d, given %d", nn.QuantBlock, n)
		}
	}
	coop := d.Coopmat()
	// The output projection is dispatched at every width this stack carries,
	// and one split has to serve them all, so it is taken at the widest —
	// which is the pass that has the fewest workgroups to spare.
	splitOut := coopSplit(dim, wideColumns, coop)
	a := &Attention{d: d, dim: dim, maxHeads: maxHeads, maxKV: maxKV,
		slotContext: slotContext, slots: slots, splitOut: splitOut, coop: coop}

	var err error
	// The scores kernel wants a wave of sixty-four where it uses the cores,
	// and the pipelines below are built at whatever the driver picks
	// otherwise; the wave is asked for per pipeline further down.
	scoresWave := uint32(0)
	if coop {
		scoresWave = coopmatWave
	}
	for _, spec := range []struct {
		into     **Pipeline
		spirv    []byte
		bindings int
		push     uintptr
		wave     uint32
	}{
		{&a.matvec, matvecSPIRV, 4, unsafe.Sizeof(moePush{}), 0},
		{&a.reduce, matmulReduceSPIRV, 2, unsafe.Sizeof(moePush{}), 0},
		{&a.prepare, attnPrepareSPIRV, 11, unsafe.Sizeof(attnPush{}), 0},
		{&a.scores, scoresSPIRV(coop), 7, unsafe.Sizeof(scorePush{}), scoresWave},
		{&a.rope, ropeTableSPIRV, 4, unsafe.Sizeof(ropePush{}), 0},
	} {
		if *spec.into, err = d.newPipeline(spec.spirv, spec.bindings, uint32(spec.push), spec.wave); err != nil {
			a.Close()
			return nil, err
		}
	}
	// The four projections read their weights once for a whole batch of
	// positions when there is one.
	if coop {
		if err := a.matvec.Wide(smallColumns, matvecWideSPIRV); err != nil {
			a.Close()
			return nil, err
		}
		for _, spec := range []struct {
			columns int
			spirv   []byte
		}{
			{tiledColumns, matmulCoop32SPIRV},
			{64, matmulCoop64SPIRV},
			{128, matmulCoop128SPIRV},
			{256, matmulCoop256SPIRV},
			{wideColumns, matmulCoop512SPIRV},
		} {
			if err := a.matvec.WideWave(spec.columns, spec.spirv, coopmatWave); err != nil {
				a.Close()
				return nil, err
			}
		}
	} else {
		for _, spec := range []struct {
			columns int
			spirv   []byte
		}{
			{smallColumns, matvecWideSPIRV},
			{tiledColumns, matmulWide32SPIRV},
			{64, matmulWide64SPIRV},
			{128, matmulWidest128SPIRV},
			{256, matmulWidest256SPIRV},
			{wideColumns, matmulWide()},
		} {
			if err := a.matvec.Wide(spec.columns, spec.spirv); err != nil {
				a.Close()
				return nil, err
			}
		}
	}

	for _, spec := range []struct {
		into  **Buffer
		size  int
		local bool
	}{
		// Everything here is device memory: the CPU writes none of it, and a
		// shader reading system memory reaches across the bus. vk/device.go's
		// Host says what that was worth.
		{&a.xq, dim * maxColumns, true},                              // the normed stream, Q8_0
		{&a.xs, 2 * dim / nn.QuantBlock * 4 * maxColumns, true},      //
		{&a.out, dim * 4 * maxColumns, true},                         // what the output projection makes
		{&a.q, maxHeads * 4 * maxColumns, true},                      // the three projections, which never leave
		{&a.k, maxKV * 4 * maxColumns, true},                         //
		{&a.v, maxKV * 4 * maxColumns, true},                         //
		{&a.qh, maxHeads * 4 * maxColumns, true},                     // the queries, rounded through fp16
		{&a.outParts, dim * 4 * maxColumns * matmulSplit(dim), true}, // its slices, when it is split
		{&a.scoreRows, 4, true},                                      // nothing: the scores never leave the workgroup, and the binding stays for the layout
		{&a.aq, maxHeads * maxColumns, true},                         // the mixed values, Q8_0
		{&a.as, 2 * maxHeads / nn.QuantBlock * 4 * maxColumns, true},
		{&a.where, maxBlocks * maxColumns * 16, false}, // per block and column: position, first, last
	} {
		var b *Buffer
		if spec.local {
			b, err = d.Local(spec.size, bufferUsageStorage)
		} else {
			b, err = d.Host(spec.size, bufferUsageStorage)
		}
		if err != nil {
			a.Close()
			return nil, err
		}
		*spec.into = b
	}
	if a.splitOut > 1 {
		if a.reduceSet, err = a.reduce.NewSet([]*Buffer{a.outParts, a.out}); err != nil {
			a.Close()
			return nil, err
		}
	}
	// One pair of tables per geometry, written once a token rather than once a
	// block: the angles depend on the position and the geometry and on nothing
	// else, which is what nn/rope.go tabulates them for.
	for i := 0; i < rotations; i++ {
		cos, err := d.Local(maxHeads*4*maxColumns, bufferUsageStorage)
		if err != nil {
			a.Close()
			return nil, err
		}
		sin, err := d.Local(maxHeads*4*maxColumns, bufferUsageStorage)
		if err != nil {
			a.Close()
			return nil, err
		}
		// The inverse frequencies are the only part of a table the CPU still
		// writes, and it writes them once: half a head of floats, against the
		// pair of tables above them, which are a megabyte and now device
		// memory because nothing on this side touches them any more.
		inv, err := d.Host(maxHeads*4, bufferUsageStorage)
		if err != nil {
			a.Close()
			return nil, err
		}
		set, err := a.rope.NewSet([]*Buffer{a.where, inv, cos, sin})
		if err != nil {
			a.Close()
			return nil, err
		}
		a.rcos = append(a.rcos, cos)
		a.rsin = append(a.rsin, sin)
		a.rinv = append(a.rinv, inv)
		a.ropeSets = append(a.ropeSets, set)
		a.ropeHalf = append(a.ropeHalf, 0)
	}
	return a, nil
}

// SetGeometry writes one rotation geometry's inverse frequencies, which is
// everything about it that does not depend on the position. It is called once,
// when the stack is built; the angles themselves are made by the card at the
// head of every pass.
//
// factors is the file's frequency factors, one per pair of dimensions, or nil
// where the geometry has none.
func (a *Attention) SetGeometry(i, dims int, base float64, factors []float32) error {
	if i < 0 || i >= len(a.rcos) {
		return fmt.Errorf("vk: rotation %d of %d", i, len(a.rcos))
	}
	if dims%2 != 0 || dims > a.maxHeads { // the frequencies take dims floats, hi and lo
		return fmt.Errorf("vk: a rotation of %d dimensions, in heads of %d", dims, a.maxHeads)
	}
	half := dims / 2
	if factors != nil && len(factors) != half {
		return fmt.Errorf("vk: %d frequency factors for %d dimensions", len(factors), dims)
	}
	// Two floats a frequency: the nearest float32 and the remainder it lost.
	// shaders/rope_table.comp says what the second one is for.
	inv := a.rinv[i].Floats()[:a.maxHeads]
	for j := 0; j < half; j++ {
		f := math.Pow(base, -2*float64(j)/float64(dims))
		if factors != nil {
			f /= float64(factors[j])
		}
		hi := float32(f)
		inv[j], inv[half+j] = hi, float32(f-float64(hi))
	}
	a.ropeHalf[i] = uint32(half)
	return nil
}

// RecordRotations fills every geometry's angle table for the columns of the
// pass about to be recorded. It goes at the head of a submission: the position
// buffer is written before it and the first block's prepare reads what it
// leaves.
func (a *Attention) RecordRotations(r *Recorder, columns int) {
	for i := range a.ropeSets {
		if a.ropeHalf[i] == 0 {
			continue
		}
		push := ropePush{halfDims: a.ropeHalf[i], stride: uint32(a.maxHeads)}
		r.Dispatch(a.ropeSets[i], uint32(columns), unsafe.Pointer(&push))
	}
}

// Input is the buffer pair a block's attention reads its normed stream from,
// so that a kernel upstream can write it instead of the caller.
func (a *Attention) Input() (*Buffer, *Buffer) { return a.xq, a.xs }

// Output is where the output projection leaves its answer.
func (a *Attention) Output() *Buffer { return a.out }

// Blocks is how many have been added.
func (a *Attention) Blocks() int { return len(a.blocks) }

// Slots is how many conversations the caches hold.
func (a *Attention) Slots() int { return a.slots }

// Bytes is what the caches take on the card, which is the part of this that
// grows with the context rather than with the model.
func (a *Attention) Bytes() int {
	n := 0
	for _, b := range a.blocks {
		if b.ck != nil {
			n += int(b.ck.size + b.cv.size)
		}
	}
	return n
}

// AddBlock uploads one block's matrices and norms, and gives it a cache — or
// the cache of the block it reads, when it computes no keys of its own. k and
// v are nil where the block has no such matrix.
func (a *Attention) AddBlock(shape BlockShape, q, k, v, o []byte, qnorm, knorm []float32) error {
	heads, kv := shape.Heads*shape.HeadDim, shape.KVHeads*shape.HeadDim
	if heads > a.maxHeads || kv > a.maxKV {
		return fmt.Errorf("vk: block %d attends over %d and %d, past the %d and %d the buffers hold",
			len(a.blocks), heads, kv, a.maxHeads, a.maxKV)
	}
	if heads%nn.QuantBlock != 0 {
		return fmt.Errorf("vk: a block's heads must come to a multiple of %d, given %d", nn.QuantBlock, heads)
	}
	if !shape.OwnsKV && (shape.KVSource < 0 || shape.KVSource >= len(a.blocks)) {
		return fmt.Errorf("vk: block %d reads block %d's cache, which is not there yet", len(a.blocks), shape.KVSource)
	}
	if shape.Capacity > a.slotContext {
		return fmt.Errorf("vk: block %d holds %d positions, past the %d one conversation was given",
			len(a.blocks), shape.Capacity, a.slotContext)
	}
	if shape.Scale == 0 {
		// A zero here would send every score to the same place and the softmax
		// would answer with a uniform mix, fluently and wrongly. It is a
		// caller that forgot the field, not a model that asked for it.
		return fmt.Errorf("vk: block %d scores at a scale of zero", len(a.blocks))
	}

	b := &attentionBlock{shape: shape}
	fail := func(err error) error {
		b.close()
		return err
	}
	var err error
	for _, spec := range []struct {
		into       **Buffer
		set        **Set
		data       []byte
		rows, cols int
		out        *Buffer
	}{
		{&b.q, &b.setQ, q, heads, a.dim, a.q},
		{&b.k, &b.setK, k, kv, a.dim, a.k},
		{&b.v, &b.setV, v, kv, a.dim, a.v},
		{&b.o, &b.setO, o, a.dim, heads, a.out},
	} {
		if spec.data == nil {
			continue
		}
		if want := spec.rows * rowBytesQ4_0(spec.cols); len(spec.data) != want {
			return fail(fmt.Errorf("vk: a projection should be %d bytes, given %d", want, len(spec.data)))
		}
		if *spec.into, err = a.d.Upload(splitQ4_0(spec.data, spec.rows, spec.cols)); err != nil {
			return fail(err)
		}
		in, scales := a.xq, a.xs
		if spec.out == a.out {
			in, scales = a.aq, a.as // the output projection reads the mix, not the stream
		}
		if *spec.set, err = a.matvec.NewSet([]*Buffer{*spec.into, in, scales, spec.out}); err != nil {
			return fail(err)
		}
		// The output projection twice: once writing the answer where the
		// kernels after it read, and once writing the slices a split product
		// makes. Which of the two a pass dispatches is its width's business.
		if spec.out == a.out && a.splitOut > 1 {
			if b.setOParts, err = a.matvec.NewSet([]*Buffer{*spec.into, in, scales, a.outParts}); err != nil {
				return fail(err)
			}
		}
	}

	if b.qnorm, err = a.d.Upload(asBytes(qnorm)); err != nil {
		return fail(err)
	}
	if b.knorm, err = a.d.Upload(asBytes(knorm)); err != nil {
		return fail(err)
	}

	// A block that computes no keys is given the buffers of the block it reads
	// from, not a copy: there is one cache and many blocks on it, which is what
	// gemma/cache.go does on the other side and for the same reason.
	var ck, cv *Buffer
	if shape.OwnsKV {
		n := a.slots * shape.Capacity * shape.KVHeads * shape.HeadDim * 2 // fp16, one ring a slot
		if b.ck, err = a.d.Local(n, bufferUsageStorage); err != nil {
			return fail(err)
		}
		if b.cv, err = a.d.Local(n, bufferUsageStorage); err != nil {
			return fail(err)
		}
		ck, cv = b.ck, b.cv
	} else {
		ck, cv = a.blocks[shape.KVSource].caches()
		if ck == nil {
			return fail(fmt.Errorf("vk: block %d reads block %d, which has no cache of its own",
				len(a.blocks), shape.KVSource))
		}
	}

	if shape.Rotation < 0 || shape.Rotation >= len(a.rcos) {
		return fail(fmt.Errorf("vk: block %d names rotation %d of %d", len(a.blocks), shape.Rotation, len(a.rcos)))
	}
	if b.setPrepare, err = a.prepare.NewSet([]*Buffer{
		a.q, a.k, a.v, b.qnorm, b.knorm, a.rcos[shape.Rotation], a.rsin[shape.Rotation], a.qh, ck, cv, a.where,
	}); err != nil {
		return fail(err)
	}
	if a.scoresFloat != nil {
		if b.setScores, err = a.scoresFloat.NewSet([]*Buffer{
			a.qh, ck, cv, a.scoreRows, a.aq, a.as, a.where, a.af,
		}); err != nil {
			return fail(err)
		}
	} else if b.setScores, err = a.scores.NewSet([]*Buffer{
		a.qh, ck, cv, a.scoreRows, a.aq, a.as, a.where,
	}); err != nil {
		return fail(err)
	}
	a.blocks = append(a.blocks, b)
	return nil
}

// caches is the pair this block reads, which is its own or an earlier one's.
func (b *attentionBlock) caches() (*Buffer, *Buffer) { return b.ck, b.cv }

// Profile is Stack.Profile, forwarded: the stamps the attention writes are
// the four products, the cache and the scores.
func (a *Attention) Profile(t *Timeline) { a.tl = t }

// SetWhere writes one block's slot, position and visible range, which is
// everything about a token that a recording cannot hold. It is what lets the
// recording be made once: the two kernels read these four numbers out of a
// buffer instead of out of the command buffer's push constants.
//
// The slot is the fourth of them and was the free one. A ring indexed by the
// position alone is a card that holds one conversation, and two clients on it
// write each other's positions — which is a wrong answer and not a slow one.
func (a *Attention) SetWhere(block, column, slot, pos, first, last int) error {
	if block < 0 || block >= maxBlocks {
		return fmt.Errorf("vk: block %d of the %d the position buffer holds", block, maxBlocks)
	}
	if column < 0 || column >= maxColumns {
		return fmt.Errorf("vk: column %d of the %d one pass carries", column, maxColumns)
	}
	if slot < 0 || slot >= a.slots {
		return fmt.Errorf("vk: slot %d of %d", slot, a.slots)
	}
	entries := unsafe.Slice((*uint32)(unsafe.Pointer(&a.where.Bytes()[0])), maxBlocks*maxColumns*4)
	at := entries[(block*maxColumns+column)*4:]
	at[0], at[1], at[2], at[3] = uint32(pos), uint32(first), uint32(last), uint32(slot)
	return nil
}

// A span is a run of a pass's columns that belong to one conversation.
//
// The scores kernel answers thirty-two columns to a workgroup and stages the
// keys of the union of their ranges, which two conversations cannot share: a
// position means a different entry of the cache in each of them. So a tile
// never straddles two, and the pass is cut into runs of one slot before the
// tiles are laid out. A prompt is one run and is tiled exactly as it was; a
// token drawn for each of four conversations is four runs of one column, which
// is the shape generation has anyway — the tile buys nothing there, since each
// conversation brings one column and its own range.
type span struct {
	first, count int
}

// spansOf cuts a pass into runs of columns that share a slot. The columns of
// one conversation are contiguous — cmd/golem-server/runner.go builds a batch
// by appending each conversation's tokens — so this is a walk and not a sort.
func spansOf(slots []int) []span {
	if len(slots) == 0 {
		return nil
	}
	out := []span{{first: 0, count: 1}}
	for c := 1; c < len(slots); c++ {
		last := &out[len(out)-1]
		if slots[c] == slots[last.first] {
			last.count++
			continue
		}
		out = append(out, span{first: c, count: 1})
	}
	return out
}

// Record puts one block's attention into a recording, for the given number of
// columns. One column is a token; more is a stretch of a prompt, and the four
// projections then read their weights once for all of them — which is the
// whole of why a prompt need not cost what the same tokens cost one at a time.
//
// runs is how the pass divides between conversations, which only the scores
// need: everything else here is per column and reads the slot out of the
// position buffer. A pass of one conversation is one run.
func (a *Attention) Record(r *Recorder, block, columns int, runs []span) {
	b := a.blocks[block]
	s := b.shape
	heads, kv := s.Heads*s.HeadDim, s.KVHeads*s.HeadDim

	project := moePush{dim: uint32(heads), ffn: uint32(a.dim), used: 1}
	kvProject := moePush{dim: uint32(kv), ffn: uint32(a.dim), used: 1}
	outProject := moePush{dim: uint32(a.dim), ffn: uint32(heads), used: 1}
	prepare := attnPush{
		heads: uint32(s.Heads), kvHeads: uint32(s.KVHeads), headDim: uint32(s.HeadDim),
		ropeDims: uint32(s.RoPEDims), block: uint32(block), capacity: uint32(s.Capacity),
		mask: uint32(ringMask(s.Capacity)), valueIsKey: boolTo(s.ValueIsKey),
		normValue: boolTo(s.NormValue), rotStride: uint32(a.maxHeads), eps: s.Eps,
	}
	score := scorePush{
		heads: uint32(s.Heads), kvHeads: uint32(s.KVHeads), headDim: uint32(s.HeadDim),
		perKV: uint32(s.Heads / s.KVHeads), block: uint32(block),
		capacity: uint32(s.Capacity), mask: uint32(ringMask(s.Capacity)),
		scale: s.Scale, // col0 and columns are the run's, set below
	}
	units := uint32(s.Heads)
	if s.OwnsKV {
		units += uint32(s.KVHeads)
	}

	// Which product answers this pass, and over how many workgroups: the
	// mat-vec writes sixteen rows to a workgroup and the tiled product
	// thirty-two, so the grid is the kernel's business and not the caller's.
	width := passWidth(columns)
	product := func(set *Set, outs int, push unsafe.Pointer) {
		if width == 1 {
			r.Dispatch(set, groups(outs), push)
			return
		}
		r.DispatchWide(set, width, a.productGroups(width, outs), push)
	}

	if b.golem != nil {
		a.recordGolemInput(r, b, columns)
	} else {
		product(b.setQ, heads, unsafe.Pointer(&project))
		if b.setK != nil {
			product(b.setK, kv, unsafe.Pointer(&kvProject))
		}
		if b.setV != nil {
			product(b.setV, kv, unsafe.Pointer(&kvProject))
		}
	}
	r.Barrier()
	a.tl.Stamp(r, "attn qkv")
	r.DispatchColumns(b.setPrepare, units, uint32(columns), unsafe.Pointer(&prepare))
	r.Barrier()
	a.tl.Stamp(r, "attn cache")
	// One workgroup to a head and a tile of scoreColumns columns. The kernel
	// keeps everything it needs in registers and shared memory, so the
	// stretches this used to be cut into — one at a time through a scratch
	// buffer the size of the deepest context — are gone with the scratch.
	for _, run := range runs {
		score.col0 = uint32(run.first)
		score.columns = uint32(run.first + run.count)
		tiles := uint32((run.count + scoreColumns - 1) / scoreColumns)
		r.DispatchColumns(b.setScores, uint32(s.Heads), tiles, unsafe.Pointer(&score))
	}
	r.Barrier()
	a.tl.Stamp(r, "attn scores")
	if b.golem != nil {
		a.recordGolemOutput(r, b, columns)
		return
	}
	if b.setOParts != nil && width >= tiledColumns {
		// The split product, and the pass that adds its slices. See
		// matmulSplit: this matrix is the stack's row-poorest, and eighty
		// workgroups do not fill the card.
		outProject.split = uint32(a.splitOut)
		r.DispatchWide(b.setOParts, width, a.productGroups(width, a.dim)*uint32(a.splitOut), unsafe.Pointer(&outProject))
		r.Barrier()
		// The slices are as wide as the binary, not as the pass: a stretch
		// shorter than the width still writes its unasked-for columns, and
		// the fold has to walk the same stride the product wrote at.
		fold := moePush{dim: uint32(a.dim), ffn: uint32(width), used: uint32(a.splitOut)}
		r.Dispatch(a.reduceSet, uint32((a.dim*width+255)/256), unsafe.Pointer(&fold))
		return
	}
	product(b.setO, a.dim, unsafe.Pointer(&outProject))
}

// ringMask is nn's rule, and gemma/cache.go's: the capacity less one when that
// is a mask, and zero when the remainder has to be taken the slow way.
func ringMask(capacity int) int {
	if capacity > 0 && capacity&(capacity-1) == 0 {
		return capacity - 1
	}
	return 0
}

func boolTo(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}

// groups is how many workgroups the matvec kernel needs for that many outputs.
func groups(outputs int) uint32 { return uint32((outputs + matvecOuts - 1) / matvecOuts) }

// productGroups is the same for whichever product a pass of that width runs.
// A tiled product is cut across its columns as well as its rows above BN of
// them, so the grid is rows over BM by columns over BN.
func (a *Attention) productGroups(width, outputs int) uint32 {
	return coopProductGroups(a.coop, width, outputs)
}

func (a *Attention) Close() {
	for _, b := range a.blocks {
		b.close()
	}
	for _, b := range []*Buffer{a.xf, a.af} {
		if b != nil {
			b.Close()
		}
	}
	a.xf, a.af = nil, nil
	a.blocks = nil
	for _, set := range a.ropeSets {
		if set != nil {
			set.Close()
		}
	}
	a.ropeSets = nil
	for _, b := range append(append([]**Buffer{}, pointers(a.rcos)...), append(append(pointers(a.rsin), pointers(a.rinv)...), []**Buffer{
		&a.where, &a.as, &a.aq, &a.scoreRows, &a.outParts, &a.qh, &a.v, &a.k, &a.q,
		&a.out, &a.xs, &a.xq,
	}...)...) {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	if a.reduceSet != nil {
		a.reduceSet.Close()
		a.reduceSet = nil
	}
	for _, p := range []**Pipeline{&a.scoresFloat, &a.scores, &a.prepare, &a.matvec, &a.reduce, &a.rope} {
		if *p != nil {
			(*p).Close()
			*p = nil
		}
	}
}

// pointers is the addresses of a slice's elements, for the close loop.
func pointers(bs []*Buffer) []**Buffer {
	out := make([]**Buffer, len(bs))
	for i := range bs {
		out[i] = &bs[i]
	}
	return out
}

func (b *attentionBlock) close() {
	b.golem.close()
	b.golem = nil
	for _, s := range []**Set{&b.setScores, &b.setPrepare, &b.setOParts, &b.setO, &b.setV, &b.setK, &b.setQ} {
		if *s != nil {
			(*s).Close()
			*s = nil
		}
	}
	for _, x := range []**Buffer{&b.cv, &b.ck, &b.knorm, &b.qnorm, &b.o, &b.v, &b.k, &b.q} {
		if *x != nil {
			(*x).Close()
			*x = nil
		}
	}
}

//go:generate glslc -O -DFLOATOUT --target-env=vulkan1.1 -fshader-stage=compute shaders/attn_scores.comp -o shaders/attn_scores_float.spv

//go:embed shaders/attn_scores_float.spv
var scoresFloatSPIRV []byte

// scoresSPIRV is the scores kernel built for this card: the block of scores on
// the matrix cores where there are any, and the scalar dot products where
// there are not.
func scoresSPIRV(coop bool) []byte {
	if coop {
		return attnScoresCoopSPIRV
	}
	return attnScoresSPIRV
}
