package vk

// The Qwen3-VL projector on the card.
//
// The tower runs once per image and not once per token, which is why it was
// left on the processor when the text model moved: a second of encoding beside
// a minute of generation is not what a picture costs. Then it was measured. A
// grid of 1040 patches through twenty-seven blocks of 1152 is 33.6 seconds
// here, because the processor's path is a matrix-vector product a patch and
// there are a thousand patches — the weights are read from memory once per
// patch and the arithmetic never reaches the ceiling. The same work as a tiled
// product on a card is a fraction of a second.
//
// Nothing of the text side is reused. The tower's weights are fp16 where the
// model's are quantized, its norms subtract a mean where the model's do not,
// its attention has no cache and no mask, and its feed forward has no gate.
// Five kernels of its own is less code than five flags on kernels that answer
// a different question.

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/vision_norm.comp -o shaders/vision_norm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/vision_gemm.comp -o shaders/vision_gemm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/vision_rope.comp -o shaders/vision_rope.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/vision_attn.comp -o shaders/vision_attn.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/vision_act.comp -o shaders/vision_act.spv

//go:embed shaders/vision_norm.spv
var visionNormSPIRV []byte

//go:embed shaders/vision_gemm.spv
var visionGemmSPIRV []byte

//go:embed shaders/vision_rope.spv
var visionRopeSPIRV []byte

//go:embed shaders/vision_attn.spv
var visionAttnSPIRV []byte

//go:embed shaders/vision_act.spv
var visionActSPIRV []byte

// visionMaxHead is the head width shaders/vision_attn.comp is written for: one
// lane an element of the accumulator, and a workgroup of that many lanes.
const visionMaxHead = 128

// VisionShape is the projector's geometry.
type VisionShape struct {
	Blocks  int
	Dim     int
	Heads   int
	HeadDim int
	FFN     int
	Merge   int // the side of the square the merger folds
	ProjDim int
	Eps     float32
	// RoPEBase is the rotation's base, which is 10000 for every Qwen-VL
	// projector and is passed rather than assumed.
	RoPEBase float32
}

func (s VisionShape) merged() int { return s.Dim * s.Merge * s.Merge }

// VisionLinearData is one projection: fp16 weights exactly as the file holds
// them, row-major with one row per output, and an F32 bias.
type VisionLinearData struct {
	W    []byte
	Bias []float32
}

// VisionBlockData is one block's weights, in the order the pass reads them.
type VisionBlockData struct {
	LN1Gain, LN1Bias []float32
	QKV, O           VisionLinearData
	LN2Gain, LN2Bias []float32
	Up, Dn           VisionLinearData
}

// VisionTailData is the last norm and the merger.
type VisionTailData struct {
	PostGain, PostBias []float32
	MM0, MM2           VisionLinearData
}

// visionMat is where one matrix lives: its bytes on this side until they are
// uploaded, where it starts inside its own group counted in halves, and where
// its bias starts in the parameter blob counted in floats.
type visionMat struct {
	src             []byte
	base            uint32
	bias            uint32
	outputs, inputs int
}

// visionNormAt is where a norm's gain and bias start in the parameter blob.
type visionNormAt struct{ gain, bias uint32 }

// visionGroup is what is uploaded as a unit: one block, or the merger. A card
// with room holds every group at once; one without holds one at a time, and
// the group is the granularity because a block's four matrices are read one
// after the other and nothing outside it is read in between.
type visionGroup struct {
	mats  []*visionMat
	bytes int
	at    int // where the group starts in the resident buffer
}

type visionBlockAt struct {
	ln1, ln2       visionNormAt
	qkv, o, up, dn visionMat
}

// A VisionPipeline is the tower resident on a device.
type VisionPipeline struct {
	d *Device
	s VisionShape

	// The parameters every pass reads: the norms' gains and biases and the
	// projections' biases, all F32 and all together. It is four hundred
	// kilobytes for the whole tower, so it is resident whatever else is.
	par    []float32
	blocks []visionBlockAt
	tail   struct {
		post     visionNormAt
		mm0, mm2 visionMat
	}
	groups []visionGroup
	built  bool

	parBuf  *Buffer
	geluBuf *Buffer
	weights *Buffer // every group, or room for the largest one
	stage   *Buffer // the streaming path's staging buffer, else nil

	normPipe, gemmPipe, ropePipe, attnPipe, actPipe *Pipeline

	sc *visionScratch

	// trace keeps every waypoint of the last Encode under llama.cpp's own
	// names for them. It is nil unless Trace was called: it submits a
	// recording a block instead of one for the tower and reads seven grids
	// back from each.
	trace map[string][]float32
}

// visionScratch is everything sized by the patch count. A second image of the
// same grid reuses it; a larger one rebuilds it.
type visionScratch struct {
	patches, tokens int

	x, nrm, qkv, q, k, mixed, tmp, up, mid, out, pos *Buffer
	xin, posin, back, tap                            *Buffer

	norm                 *Set
	gQKV, gO, gUp, gDn   *Set
	gMM0, gMM2           *Set
	rope, attn           *Set
	add, geluUp, geluMid *Set
}

type visionNormPush struct {
	dim, gain, bias uint32
	eps             float32
}

type visionGemmPush struct {
	outputs, inputs, patches, base, bias uint32
}

type visionRopePush struct {
	dim, heads, head, sect uint32
	base                   float32
}

type visionAttnPush struct {
	patches, dim, head uint32
	scale              float32
}

type visionActPush struct{ count, op uint32 }

// NewVisionPipeline creates the tower's kernels. The weights arrive block by
// block afterwards and nothing reaches the card until Prepare.
func NewVisionPipeline(d *Device, s VisionShape) (*VisionPipeline, error) {
	if s.HeadDim > visionMaxHead {
		return nil, fmt.Errorf("vk: the vision attention is written for heads of at most %d, this one is %d",
			visionMaxHead, s.HeadDim)
	}
	if s.HeadDim%4 != 0 {
		return nil, fmt.Errorf("vk: the vision rotation needs four sections in a head, and %d does not divide by four", s.HeadDim)
	}
	if s.Dim%2 != 0 || s.FFN%2 != 0 || s.merged()%2 != 0 {
		return nil, fmt.Errorf("vk: an fp16 row is read two halves a word, so every width must be even")
	}
	p := &VisionPipeline{d: d, s: s}

	var err error
	if p.normPipe, err = d.NewPipeline(visionNormSPIRV, 3, uint32(unsafe.Sizeof(visionNormPush{}))); err != nil {
		return nil, err
	}
	if p.gemmPipe, err = d.NewPipeline(visionGemmSPIRV, 4, uint32(unsafe.Sizeof(visionGemmPush{}))); err != nil {
		p.Close()
		return nil, err
	}
	if p.ropePipe, err = d.NewPipeline(visionRopeSPIRV, 4, uint32(unsafe.Sizeof(visionRopePush{}))); err != nil {
		p.Close()
		return nil, err
	}
	if p.attnPipe, err = d.NewPipeline(visionAttnSPIRV, 4, uint32(unsafe.Sizeof(visionAttnPush{}))); err != nil {
		p.Close()
		return nil, err
	}
	if p.actPipe, err = d.NewPipeline(visionActSPIRV, 3, uint32(unsafe.Sizeof(visionActPush{}))); err != nil {
		p.Close()
		return nil, err
	}
	return p, nil
}

// keepNorm puts a gain and a bias in the parameter blob and says where they
// landed.
func (p *VisionPipeline) keepNorm(gain, bias []float32, width int) (visionNormAt, error) {
	if len(gain) != width || len(bias) != width {
		return visionNormAt{}, fmt.Errorf("vk: a norm of %d wants a gain and a bias of that width, given %d and %d",
			width, len(gain), len(bias))
	}
	at := visionNormAt{gain: uint32(len(p.par))}
	p.par = append(p.par, gain...)
	at.bias = uint32(len(p.par))
	p.par = append(p.par, bias...)
	return at, nil
}

// keepLinear checks one projection and puts its bias in the blob. The weights
// stay where they are until Prepare.
func (p *VisionPipeline) keepLinear(l VisionLinearData, outputs, inputs int) (visionMat, error) {
	if want := outputs * inputs * 2; len(l.W) != want {
		return visionMat{}, fmt.Errorf("vk: an fp16 matrix of %d by %d is %d bytes, given %d",
			outputs, inputs, want, len(l.W))
	}
	if len(l.Bias) != outputs {
		return visionMat{}, fmt.Errorf("vk: a projection onto %d wants that many biases, given %d", outputs, len(l.Bias))
	}
	m := visionMat{src: l.W, outputs: outputs, inputs: inputs, bias: uint32(len(p.par))}
	p.par = append(p.par, l.Bias...)
	return m, nil
}

// AddBlock takes one block's weights, in order.
func (p *VisionPipeline) AddBlock(b VisionBlockData) error {
	if p.built {
		return fmt.Errorf("vk: the vision tower is already on the card")
	}
	if len(p.blocks) >= p.s.Blocks {
		return fmt.Errorf("vk: the tower has %d blocks and a %dth was given", p.s.Blocks, len(p.blocks)+1)
	}
	s := p.s
	var at visionBlockAt
	var err error
	if at.ln1, err = p.keepNorm(b.LN1Gain, b.LN1Bias, s.Dim); err != nil {
		return err
	}
	if at.qkv, err = p.keepLinear(b.QKV, 3*s.Dim, s.Dim); err != nil {
		return err
	}
	if at.o, err = p.keepLinear(b.O, s.Dim, s.Dim); err != nil {
		return err
	}
	if at.ln2, err = p.keepNorm(b.LN2Gain, b.LN2Bias, s.Dim); err != nil {
		return err
	}
	if at.up, err = p.keepLinear(b.Up, s.FFN, s.Dim); err != nil {
		return err
	}
	if at.dn, err = p.keepLinear(b.Dn, s.Dim, s.FFN); err != nil {
		return err
	}
	p.blocks = append(p.blocks, at)
	return nil
}

// SetTail takes the tower's last norm and the merger.
func (p *VisionPipeline) SetTail(t VisionTailData) error {
	if p.built {
		return fmt.Errorf("vk: the vision tower is already on the card")
	}
	s := p.s
	var err error
	if p.tail.post, err = p.keepNorm(t.PostGain, t.PostBias, s.Dim); err != nil {
		return err
	}
	if p.tail.mm0, err = p.keepLinear(t.MM0, s.merged(), s.merged()); err != nil {
		return err
	}
	if p.tail.mm2, err = p.keepLinear(t.MM2, s.ProjDim, s.merged()); err != nil {
		return err
	}
	return nil
}

// Prepare lays the weights out, decides where they live and uploads what can
// live there.
//
// The Qwen3-VL tower is 884 mebibytes in fp16 and it shares a card with the
// text model, which for a 27B at Q4_0 on sixteen gigabytes leaves it a margin
// that a long context spends. So the resident form is tried and its failure is
// not an error: what follows is the same recording against a buffer the size
// of one group, refilled from the host before each. That costs the tower's
// bytes across the bus once an image — 1.41 seconds against 1.03 — which is
// the right side of the thirty-three the processor took.
func (p *VisionPipeline) Prepare() error {
	if p.built {
		return nil
	}
	if len(p.blocks) != p.s.Blocks {
		return fmt.Errorf("vk: the tower has %d blocks and %d were given", p.s.Blocks, len(p.blocks))
	}
	if p.tail.mm2.src == nil {
		return fmt.Errorf("vk: the tower's merger was not given")
	}

	// The layout: one group a block, then the merger. A matrix's base is
	// counted in halves and from the start of its own group, which is what
	// lets the same push constant address a resident buffer and a staged one.
	p.groups = nil
	for i := range p.blocks {
		b := &p.blocks[i]
		p.groups = append(p.groups, p.layout(&b.qkv, &b.o, &b.up, &b.dn))
	}
	p.groups = append(p.groups, p.layout(&p.tail.mm0, &p.tail.mm2))

	total, widest := 0, 0
	for i := range p.groups {
		p.groups[i].at = total
		total += p.groups[i].bytes
		widest = max(widest, p.groups[i].bytes)
	}

	var err error
	if p.parBuf, err = p.d.Upload(asBytes(p.par)); err != nil {
		return err
	}
	if p.geluBuf, err = p.d.Upload(asBytes(nn.GELUTableData())); err != nil {
		return err
	}

	// Resident first. A refusal here is the card saying it has no room, and
	// the streaming path is the answer rather than the failure.
	//
	// GOLEM_VISION_STREAM skips the attempt. Which of the two paths a machine
	// takes depends on what else is on its card, so the one it does not take
	// would otherwise never be run — and a path that is never run is a path
	// that is wrong.
	if p.weights, err = p.tryResident(total); err == nil {
		if err = p.uploadGroups(); err == nil {
			p.built = true
			return nil
		}
		p.weights.Close()
		p.weights = nil
	}

	if p.weights, err = p.d.Local(widest, bufferUsageStorage|bufferUsageTransferDst); err != nil {
		return fmt.Errorf("vk: the card has room for neither the whole tower (%d MiB) nor one of its groups (%d MiB): %w",
			total>>20, widest>>20, err)
	}
	if p.stage, err = p.d.Host(widest, bufferUsageTransferSrc); err != nil {
		return err
	}
	p.built = true
	return nil
}

// tryResident allocates room for the whole tower, unless the environment says
// not to.
func (p *VisionPipeline) tryResident(total int) (*Buffer, error) {
	if os.Getenv("GOLEM_VISION_STREAM") != "" {
		return nil, fmt.Errorf("vk: GOLEM_VISION_STREAM asks for the streaming path")
	}
	return p.d.Local(total, bufferUsageStorage|bufferUsageTransferDst)
}

// layout assigns each matrix its place inside one group.
func (p *VisionPipeline) layout(mats ...*visionMat) visionGroup {
	g := visionGroup{mats: mats}
	for _, m := range mats {
		m.base = uint32(g.bytes / 2)
		g.bytes += len(m.src)
	}
	return g
}

// uploadGroups fills the resident buffer, a bounded staging chunk at a time.
func (p *VisionPipeline) uploadGroups() error {
	const chunk = 64 << 20
	stage, err := p.d.Host(chunk, bufferUsageTransferSrc)
	if err != nil {
		return err
	}
	defer stage.Close()
	for _, g := range p.groups {
		for _, m := range g.mats {
			at := g.at + int(m.base)*2
			for off := 0; off < len(m.src); off += chunk {
				n := min(chunk, len(m.src)-off)
				copy(stage.Bytes(), m.src[off:off+n])
				if err := p.d.Submit(func(r *Recorder) { r.Copy(p.weights, at+off, stage, n) }); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// Resident reports whether the whole tower is on the card, as against one
// group at a time across the bus.
func (p *VisionPipeline) Resident() bool { return p.built && p.stage == nil }

// Trace makes the next Encode keep every waypoint, under the names
// models/qwen3vl.cpp gives its nodes. It submits a recording a block rather
// than one for the tower and reads seven grids back from each, which is why it
// is for the reference tests and not for a running model.
func (p *VisionPipeline) Trace() { p.trace = map[string][]float32{} }

// Waypoint is what the last traced Encode left under that name, or nil.
func (p *VisionPipeline) Waypoint(name string) []float32 { return p.trace[name] }

// The seven grids a traced block keeps, in the order the pass reaches them.
var visionTaps = []string{
	"ln1", "Qcur_rope", "Kcur_rope", "attn_out", "ffn_inp", "ffn_inp_normed", "layer_out",
}

// Encode runs the tower over one image.
//
// xs is the patch grid the convolutions and the learned positions already
// made, patch-major and Dim wide; pos is each patch's row and column, the two
// interleaved. out receives one row of ProjDim per merged group of patches.
//
// The patch embedding stays on the processor. It is one product of 768 inputs
// a patch against a matrix of 1152 rows — two per cent of what a block costs,
// times one rather than twenty-seven — and it is entangled with the resize,
// the reordering and the antialiased interpolation of the learned table, none
// of which is arithmetic a card is wanted for.
func (p *VisionPipeline) Encode(xs []float32, pos []uint32, out []float32) error {
	if !p.built {
		return fmt.Errorf("vk: the vision tower has not been prepared")
	}
	s := p.s
	group := s.Merge * s.Merge
	patches := len(xs) / s.Dim
	if patches*s.Dim != len(xs) {
		return fmt.Errorf("vk: %d values are not a whole number of patches of %d", len(xs), s.Dim)
	}
	if patches%group != 0 {
		return fmt.Errorf("vk: %d patches do not divide into squares of %d", patches, group)
	}
	if len(pos) != 2*patches {
		return fmt.Errorf("vk: %d patches want %d positions, given %d", patches, 2*patches, len(pos))
	}
	tokens := patches / group
	if len(out) != tokens*s.ProjDim {
		return fmt.Errorf("vk: %d patches make %d rows of %d, given room for %d values",
			patches, tokens, s.ProjDim, len(out))
	}

	if err := p.scratchFor(patches); err != nil {
		return err
	}
	sc := p.sc

	copy(sc.xin.Floats(), xs)
	copy(unsafe.Slice((*uint32)(unsafe.Pointer(&sc.posin.Bytes()[0])), len(pos)), pos)
	if err := p.d.Submit(func(r *Recorder) {
		r.Copy(sc.x, 0, sc.xin, len(xs)*4)
		r.Copy(sc.pos, 0, sc.posin, len(pos)*4)
	}); err != nil {
		return err
	}

	for i := range p.blocks {
		if err := p.d.Submit(func(r *Recorder) {
			p.stream(r, i)
			p.recordBlock(r, i, patches)
		}); err != nil {
			return fmt.Errorf("vk: the vision tower failed at block %d: %w", i, err)
		}
		if p.trace != nil {
			p.harvest(i, patches)
		}
	}

	if err := p.d.Submit(func(r *Recorder) {
		p.stream(r, len(p.blocks))
		p.recordTail(r, patches, tokens)
		r.Barrier()
		r.Copy(sc.back, 0, sc.out, tokens*s.ProjDim*4)
	}); err != nil {
		return fmt.Errorf("vk: the vision tower failed at the merger: %w", err)
	}
	copy(out, sc.back.Floats()[:tokens*s.ProjDim])
	return nil
}

// stream puts one group's weights where the recording expects them. On a card
// that holds the whole tower it does nothing, and the group's own place in the
// resident buffer is what the push constants carry.
func (p *VisionPipeline) stream(r *Recorder, group int) {
	if p.stage == nil {
		return
	}
	g := p.groups[group]
	at := 0
	for _, m := range g.mats {
		copy(p.stage.Bytes()[at:], m.src)
		at += len(m.src)
	}
	r.Copy(p.weights, 0, p.stage, g.bytes)
	r.Barrier()
}

// base is where a matrix starts in the weight buffer, in halves — which is
// inside its group when a group is all the card holds, and inside the whole
// tower when the whole tower is on it.
func (p *VisionPipeline) base(group int, m *visionMat) uint32 {
	if p.stage != nil {
		return m.base
	}
	return uint32(p.groups[group].at/2) + m.base
}

func (p *VisionPipeline) norm(r *Recorder, at visionNormAt, patches int) {
	push := visionNormPush{dim: uint32(p.s.Dim), gain: at.gain, bias: at.bias, eps: p.s.Eps}
	r.Dispatch(p.sc.norm, uint32(patches), unsafe.Pointer(&push))
}

func (p *VisionPipeline) gemm(r *Recorder, set *Set, group int, m *visionMat, columns int) {
	push := visionGemmPush{
		outputs: uint32(m.outputs), inputs: uint32(m.inputs), patches: uint32(columns),
		base: p.base(group, m), bias: m.bias,
	}
	const bm, bn = 64, 64
	groups := uint32(((m.outputs + bm - 1) / bm) * ((columns + bn - 1) / bn))
	r.Dispatch(set, groups, unsafe.Pointer(&push))
}

func (p *VisionPipeline) act(r *Recorder, set *Set, count int, op uint32) {
	push := visionActPush{count: uint32(count), op: op}
	r.Dispatch(set, uint32((count+255)/256), unsafe.Pointer(&push))
}

// tap keeps one waypoint of a traced block.
func (p *VisionPipeline) tap(r *Recorder, name string, src *Buffer, patches int) {
	if p.trace == nil {
		return
	}
	for i, n := range visionTaps {
		if n == name {
			r.Barrier()
			r.Copy(p.sc.tap, i*patches*p.s.Dim*4, src, patches*p.s.Dim*4)
			return
		}
	}
	panic("vk: " + name + " is not a vision waypoint")
}

// recordBlock is one block of the tower: a norm, the fused projection, the
// rotation, full attention, the output projection and a residual; then a
// second norm, a gateless feed forward under GELU, and a second residual.
func (p *VisionPipeline) recordBlock(r *Recorder, i, patches int) {
	s, sc, b := p.s, p.sc, &p.blocks[i]

	p.norm(r, b.ln1, patches)
	p.tap(r, "ln1", sc.nrm, patches)
	r.Barrier()

	p.gemm(r, sc.gQKV, i, &b.qkv, patches)
	r.Barrier()

	rope := visionRopePush{
		dim: uint32(s.Dim), heads: uint32(s.Heads), head: uint32(s.HeadDim),
		sect: uint32(s.HeadDim / 4), base: s.RoPEBase,
	}
	r.Dispatch(sc.rope, uint32(patches), unsafe.Pointer(&rope))
	p.tap(r, "Qcur_rope", sc.q, patches)
	p.tap(r, "Kcur_rope", sc.k, patches)
	r.Barrier()

	attn := visionAttnPush{
		patches: uint32(patches), dim: uint32(s.Dim), head: uint32(s.HeadDim),
		scale: float32(1 / math.Sqrt(float64(s.HeadDim))),
	}
	r.DispatchColumns(sc.attn, uint32(patches), uint32(s.Heads), unsafe.Pointer(&attn))
	r.Barrier()

	p.gemm(r, sc.gO, i, &b.o, patches)
	p.tap(r, "attn_out", sc.tmp, patches)
	r.Barrier()

	p.act(r, sc.add, patches*s.Dim, 1)
	p.tap(r, "ffn_inp", sc.x, patches)
	r.Barrier()

	p.norm(r, b.ln2, patches)
	p.tap(r, "ffn_inp_normed", sc.nrm, patches)
	r.Barrier()

	p.gemm(r, sc.gUp, i, &b.up, patches)
	r.Barrier()

	p.act(r, sc.geluUp, patches*s.FFN, 0)
	r.Barrier()

	p.gemm(r, sc.gDn, i, &b.dn, patches)
	r.Barrier()

	p.act(r, sc.add, patches*s.Dim, 1)
	p.tap(r, "layer_out", sc.x, patches)
}

// recordTail is the tower's last norm and the merger.
//
// The merger is a concatenation and two projections with a GELU between them,
// and the concatenation is free: the patch order was arranged so that the four
// patches of a square are four consecutive rows, and four consecutive rows of
// Dim are one row of Dim*Merge*Merge already.
func (p *VisionPipeline) recordTail(r *Recorder, patches, tokens int) {
	s, sc := p.s, p.sc
	p.norm(r, p.tail.post, patches)
	r.Barrier()
	p.gemm(r, sc.gMM0, len(p.blocks), &p.tail.mm0, tokens)
	r.Barrier()
	p.act(r, sc.geluMid, tokens*s.merged(), 0)
	r.Barrier()
	p.gemm(r, sc.gMM2, len(p.blocks), &p.tail.mm2, tokens)
}

// harvest reads one traced block's seven grids back.
func (p *VisionPipeline) harvest(block, patches int) {
	n := patches * p.s.Dim
	all := p.sc.tap.Floats()
	for i, name := range visionTaps {
		p.trace[fmt.Sprintf("%s-%d", name, block)] = append([]float32(nil), all[i*n:(i+1)*n]...)
	}
}

// scratchFor makes the buffers this many patches need, or keeps the ones a
// previous image of the same size left.
func (p *VisionPipeline) scratchFor(patches int) error {
	// A traced pass needs the tap buffer, and scratch left by an untraced one
	// of the same size does not have it.
	if p.sc != nil && p.sc.patches == patches && (p.trace == nil || p.sc.tap != nil) {
		return nil
	}
	if p.sc != nil {
		p.sc.close()
		p.sc = nil
	}
	s := p.s
	tokens := patches / (s.Merge * s.Merge)
	sc := &visionScratch{patches: patches, tokens: tokens}

	local := func(b **Buffer, floats int) error {
		v, err := p.d.Local(floats*4, bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst)
		*b = v
		return err
	}
	for _, c := range []struct {
		b      **Buffer
		floats int
	}{
		{&sc.x, patches * s.Dim},
		{&sc.nrm, patches * s.Dim},
		{&sc.qkv, patches * 3 * s.Dim},
		{&sc.q, patches * s.Dim},
		{&sc.k, patches * s.Dim},
		{&sc.mixed, patches * s.Dim},
		{&sc.tmp, patches * s.Dim},
		{&sc.up, patches * s.FFN},
		{&sc.mid, tokens * s.merged()},
		{&sc.out, tokens * s.ProjDim},
		{&sc.pos, 2 * patches},
	} {
		if err := local(c.b, c.floats); err != nil {
			sc.close()
			return fmt.Errorf("vk: the card has no room for a grid of %d patches: %w", patches, err)
		}
	}
	var err error
	if sc.xin, err = p.d.Host(patches*s.Dim*4, bufferUsageTransferSrc); err != nil {
		sc.close()
		return err
	}
	if sc.posin, err = p.d.Host(2*patches*4, bufferUsageTransferSrc); err != nil {
		sc.close()
		return err
	}
	if sc.back, err = p.d.Readback(tokens*s.ProjDim*4, bufferUsageTransferDst); err != nil {
		sc.close()
		return err
	}
	if p.trace != nil {
		if sc.tap, err = p.d.Readback(len(visionTaps)*patches*s.Dim*4, bufferUsageTransferDst); err != nil {
			sc.close()
			return err
		}
	}

	for _, c := range []struct {
		set     **Set
		pipe    *Pipeline
		buffers []*Buffer
	}{
		{&sc.norm, p.normPipe, []*Buffer{sc.x, p.parBuf, sc.nrm}},
		{&sc.gQKV, p.gemmPipe, []*Buffer{p.weights, sc.nrm, p.parBuf, sc.qkv}},
		{&sc.gO, p.gemmPipe, []*Buffer{p.weights, sc.mixed, p.parBuf, sc.tmp}},
		{&sc.gUp, p.gemmPipe, []*Buffer{p.weights, sc.nrm, p.parBuf, sc.up}},
		{&sc.gDn, p.gemmPipe, []*Buffer{p.weights, sc.up, p.parBuf, sc.tmp}},
		{&sc.gMM0, p.gemmPipe, []*Buffer{p.weights, sc.nrm, p.parBuf, sc.mid}},
		{&sc.gMM2, p.gemmPipe, []*Buffer{p.weights, sc.mid, p.parBuf, sc.out}},
		{&sc.rope, p.ropePipe, []*Buffer{sc.qkv, sc.pos, sc.q, sc.k}},
		{&sc.attn, p.attnPipe, []*Buffer{sc.q, sc.k, sc.qkv, sc.mixed}},
		{&sc.add, p.actPipe, []*Buffer{sc.x, sc.tmp, p.geluBuf}},
		{&sc.geluUp, p.actPipe, []*Buffer{sc.up, sc.up, p.geluBuf}},
		{&sc.geluMid, p.actPipe, []*Buffer{sc.mid, sc.mid, p.geluBuf}},
	} {
		set, err := c.pipe.NewSet(c.buffers)
		if err != nil {
			sc.close()
			return err
		}
		*c.set = set
	}

	p.sc = sc
	return nil
}

func (sc *visionScratch) close() {
	for _, s := range []**Set{&sc.norm, &sc.gQKV, &sc.gO, &sc.gUp, &sc.gDn, &sc.gMM0, &sc.gMM2,
		&sc.rope, &sc.attn, &sc.add, &sc.geluUp, &sc.geluMid} {
		if *s != nil {
			(*s).Close()
			*s = nil
		}
	}
	for _, b := range []**Buffer{&sc.x, &sc.nrm, &sc.qkv, &sc.q, &sc.k, &sc.mixed, &sc.tmp,
		&sc.up, &sc.mid, &sc.out, &sc.pos, &sc.xin, &sc.posin, &sc.back, &sc.tap} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}

// Bytes is what the tower holds in device memory, the scratch included.
func (p *VisionPipeline) Bytes() int {
	n := 0
	for _, b := range []*Buffer{p.parBuf, p.geluBuf, p.weights} {
		if b != nil {
			n += b.Size()
		}
	}
	if p.sc != nil {
		for _, b := range []*Buffer{p.sc.x, p.sc.nrm, p.sc.qkv, p.sc.q, p.sc.k, p.sc.mixed,
			p.sc.tmp, p.sc.up, p.sc.mid, p.sc.out, p.sc.pos} {
			if b != nil {
				n += b.Size()
			}
		}
	}
	return n
}

func (p *VisionPipeline) Close() {
	if p.sc != nil {
		p.sc.close()
		p.sc = nil
	}
	for _, pipe := range []**Pipeline{&p.normPipe, &p.gemmPipe, &p.ropePipe, &p.attnPipe, &p.actPipe} {
		if *pipe != nil {
			(*pipe).Close()
			*pipe = nil
		}
	}
	for _, b := range []**Buffer{&p.parBuf, &p.geluBuf, &p.weights, &p.stage} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	p.built = false
}

// VisionWaypoints are the names a traced Encode keeps a grid under, which are
// models/qwen3vl.cpp's own names for its nodes. Each is suffixed with the
// block it came from.
func VisionWaypoints() []string { return visionTaps }
