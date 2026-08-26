package vk

// Every block of a model, run in one submission.
//
// This is the last of the four moves and the one the other three were for. A
// submission costs sixty-three microseconds whatever is in it, and a card
// handed a hundred microseconds of work and then left alone runs at half its
// clocks: measured, the same feed-forward work takes 306 microseconds a block
// submitted on its own and 117 when it is not allowed to rest. Sixty
// submissions a token was most of what a token cost, and none of it was
// arithmetic.
//
// So the arithmetic between the matrices comes over too — the norms, the
// residual adds, the routing — not because any of it is expensive but because
// as long as one of them was here, the block had to come back to have it done.
// What crosses now is the embedding in and the last hidden state out, once
// each.
//
// The logit head stays a submission of its own. Its activation is Q8_K, which
// is a different quantizer from the Q8_0 everything else uses, and a second
// submission a token costs sixty-three microseconds against the thousands this
// saves.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/norm.comp -o shaders/norm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/router_logits.comp -o shaders/router_logits.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/router_pick.comp -o shaders/router_pick.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/combine.comp -o shaders/combine.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/embed_q6k.comp -o shaders/embed_q6k.spv

//go:embed shaders/norm.spv
var normSPIRV []byte

//go:embed shaders/router_logits.spv
var routerLogitsSPIRV []byte

//go:embed shaders/router_pick.spv
var routerPickSPIRV []byte

//go:embed shaders/combine.spv
var combineSPIRV []byte

//go:embed shaders/embed_q6k.spv
var embedQ6KSPIRV []byte

// The flags shaders/norm.comp reads.
const (
	normAdd    = 1 << iota // the input is a + b rather than a
	normSum                // write that sum, which is the block's residual
	normGain               // multiply by a gain vector
	normVScale             // and by a second one, which only the router wants
	normFloat              // write the normed vector as floats
	normQuant              // write its Q8_0 form
)

// normPush is shaders/norm.comp's block.
type normPush struct {
	n      uint32
	flags  uint32
	eps    float32
	scalar float32
}

// pickLanes is shaders/router_pick.comp's workgroup, which is one lane per
// expert and so also the most experts it can choose among.
const pickLanes = 256

// routerBN is how many columns shaders/router_logits.comp stages in one workgroup.
const routerBN = 8

// routerPush is shaders/router.comp's.
type routerPush struct {
	n       uint32
	experts uint32
	used    uint32
	columns uint32
	eps     float32
	scalar  float32
}

// embedPush is shaders/embed_q6k.comp's.
type embedPush struct {
	cols        uint32
	superblocks uint32
	scale       float32
}

// combinePush is shaders/combine.comp's.
type combinePush struct {
	n        uint32
	eps      float32
	outScale float32
	layout   uint32
}

// The three shapes a block's end can have, which is the whole of what differs
// between the architectures this stack runs.
const (
	// LayoutMixture is Gemma 4's mixture block: the attention normed on the way
	// out, two feed-forward branches, and three post-norms.
	LayoutMixture = iota
	// LayoutDense is Gemma 4's dense block: the same, with one branch and one
	// post-norm.
	LayoutDense
	// LayoutPreNorm is the ordinary transformer block, which is what Qwen3 has:
	// norm, attend, add; norm, feed forward, add. No norm on the way out of
	// either half, and no scalar over the block.
	LayoutPreNorm
)

// A Stack is a model's blocks, their norms and their router, over an Attention
// and a Mixture that hold the matrices.
type Stack struct {
	d   *Device
	dim int
	eps float32

	attn *Attention
	mix  *Mixture

	norm, routerW, routerPick, combine *Pipeline
	// embed reads one row of the token embedding into the stream, when the
	// caller gave the stack a table to read it from.
	embed    *Pipeline
	embedSet *Set
	ids      *Buffer // one identifier a column, negative where the caller wrote its own
	embedOf  embedPush

	routerIn  *Buffer // the residual under the router's own norm and scale
	routerOut *Buffer // one logit per expert

	// outGain is the model's final norm, when the caller gave it one. The
	// last thing a pass does is then the same kernel as every other norm,
	// reading the stream and writing it back in place — which is what lets a
	// caller read the hidden state out already normed and lets a step of this
	// engine be compared against llama.cpp's at the same point.
	outGain  *Buffer
	setFinal *Set

	// xs is the stream the blocks read and write, and it lives in device
	// memory. It was allocated host-visible for years because the caller
	// seeds it and reads it back, and that put the single hottest buffer in
	// the model across the bus: three of a block's stages touch it — the
	// attention norm reads it, the residual norm reads it again, the combine
	// writes it — and all three measured six to seven gigabytes a second on a
	// card that gives six hundred and forty, whatever was done to their
	// kernels. That is the forty-times figure the norm's header records, and
	// it was never the norm. stage is the host's copy of it, moved in and out
	// by two DMA copies at the ends of the recording.
	xs     *Buffer
	stage  *Buffer // the host side of xs, one copy in and one copy out a pass
	resid  *Buffer // between the two halves of a block
	normed *Buffer // the attention's output under its post-norm
	none   *Buffer // bound where a shader's optional input is unused

	// traces keeps each block's output, which the recording would otherwise
	// overwrite on its way to the next. It is a copy of eleven kilobytes a
	// block against a token's several milliseconds, and it is what lets the
	// reference tests still name the block a divergence begins in — which is
	// the instrument this engine was built with.
	traces *Buffer

	blocks []*stackBlock
	trace  bool // whether Run keeps each block's output; see traces
	tl     *Timeline

	// programs are the recordings, one per width of pass, made the first time
	// that width is asked for and kept. A token is one column and a prompt is
	// as many as vk.Columns allows, so a conversation makes two. Trace and
	// Profile change what is recorded, so both drop them all.
	programs map[int]*Program
}

// A stackBlock is one block's norms, its router, and the bindings that read
// them. The matrices are the Attention's and the Mixture's.
type stackBlock struct {
	gains    []*Buffer // seven of them, in the order the sets below bind them
	routerW  *Buffer
	routerV  *Buffer
	downs    *Buffer
	outScale float32
	layout   int

	setAttnNorm *Set // the stream under the attention norm, quantized
	setPostAttn *Set // the attention's answer under its post-norm
	setResid    *Set // the residual, and the shared branch's input on top of it
	setExpert   *Set // the expert branch's input
	setRouterIn *Set // the router's input, which is the residual under its own norm
	setRouterW  *Set
	setPick     *Set
	setCombine  *Set
}

// NewStack builds the kernels and the buffers a token passes through. The
// Attention and the Mixture are taken over: closing the stack closes them.
func NewStack(d *Device, dim int, eps float32, attn *Attention, mix *Mixture) (*Stack, error) {
	s := &Stack{d: d, dim: dim, eps: eps, attn: attn, mix: mix}

	var err error
	for _, spec := range []struct {
		into     **Pipeline
		spirv    []byte
		bindings int
		push     uintptr
	}{
		{&s.norm, normSPIRV, 8, unsafe.Sizeof(normPush{})},
		{&s.routerW, routerLogitsSPIRV, 3, unsafe.Sizeof(routerPush{})},
		{&s.routerPick, routerPickSPIRV, 4, unsafe.Sizeof(routerPush{})},
		{&s.combine, combineSPIRV, 7, unsafe.Sizeof(combinePush{})},
		{&s.embed, embedQ6KSPIRV, 3, unsafe.Sizeof(embedPush{})},
	} {
		if *spec.into, err = d.NewPipeline(spec.spirv, spec.bindings, uint32(spec.push)); err != nil {
			s.Close()
			return nil, err
		}
	}

	if s.xs, err = d.Local(dim*4*maxColumns, bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst); err != nil {
		s.Close()
		return nil, err
	}
	if s.stage, err = d.Readback(dim*4*maxColumns, bufferUsageTransferSrc|bufferUsageTransferDst); err != nil {
		s.Close()
		return nil, err
	}
	for _, into := range []**Buffer{&s.resid, &s.normed, &s.none, &s.routerIn} {
		if *into, err = d.Local(dim*4*maxColumns, bufferUsageStorage); err != nil {
			s.Close()
			return nil, err
		}
	}
	return s, nil
}

// SetEmbedding gives the stack the token embedding to read, which is the
// tensor the logit head already holds: pass its Table. A pass then looks its
// own row up rather than being handed one, and SetTokens says which.
//
// It may be called at any time; the recordings are dropped and made again.
func (s *Stack) SetEmbedding(table *Buffer, cols int, scale float32) error {
	if cols != s.dim {
		return fmt.Errorf("vk: the embedding is %d wide and the stream is %d", cols, s.dim)
	}
	if cols%256 != 0 {
		return fmt.Errorf("vk: a Q6_K row needs a multiple of 256 columns, given %d", cols)
	}
	if s.embedSet != nil {
		return fmt.Errorf("vk: the embedding is already set")
	}
	var err error
	if s.ids, err = s.d.Host(maxColumns*4, bufferUsageStorage); err != nil {
		return err
	}
	if s.embedSet, err = s.embed.NewSet([]*Buffer{table, s.ids, s.xs}); err != nil {
		return err
	}
	s.embedOf = embedPush{cols: uint32(cols), superblocks: uint32(cols / 256), scale: scale}
	s.programs = nil
	return nil
}

// Embedding says whether the stack looks its own embedding up.
func (s *Stack) Embedding() bool { return s.embedSet != nil }

// SetTokens says which row of the embedding each column of the pass reads. A
// negative identifier leaves that column of the stream alone, which is how a
// picture reaches the model: the vision tower's rows are written there by the
// caller and nothing must overwrite them.
func (s *Stack) SetTokens(ids []int32) error {
	if s.embedSet == nil {
		return fmt.Errorf("vk: this stack has no embedding to look up")
	}
	if len(ids) > maxColumns {
		return fmt.Errorf("vk: a pass carries %d columns, given %d identifiers", maxColumns, len(ids))
	}
	copy(unsafe.Slice((*int32)(unsafe.Pointer(&s.ids.Bytes()[0])), maxColumns), ids)
	return nil
}

// SetOutputNorm puts the model's final norm on the card. Without it a pass
// leaves the last block's residual in the stream and the caller norms it
// itself; with it the stream comes back normed.
//
// It is called once, before Ready, and the gain is uploaded rather than kept.
func (s *Stack) SetOutputNorm(gain []float32) error {
	if len(gain) != s.dim {
		return fmt.Errorf("vk: the final norm should be %d wide, given %d", s.dim, len(gain))
	}
	if s.outGain != nil {
		return fmt.Errorf("vk: the final norm is already set")
	}
	buf, err := s.d.Upload(asBytes(gain))
	if err != nil {
		return err
	}
	s.outGain = buf
	// a, b, gain, vscale, sum, y, yq, ys — in place, and nothing quantized.
	set, err := s.norm.NewSet([]*Buffer{s.xs, s.none, s.outGain, s.none, s.none, s.xs, s.none, s.none})
	if err != nil {
		return err
	}
	s.setFinal = set
	s.programs = nil
	return nil
}

// OutputNorm says whether the final norm is on the card.
func (s *Stack) OutputNorm() bool { return s.setFinal != nil }

// SetGeometry writes one rotation geometry's inverse frequencies. It is called
// once, when the stack is built: the angles themselves are made by the card at
// the head of every pass, out of the position buffer.
func (s *Stack) SetGeometry(i, dims int, base float64, factors []float32) error {
	return s.attn.SetGeometry(i, dims, base, factors)
}

// Columns is how many positions one pass may carry.
func (s *Stack) Columns() int { return maxColumns }

// Stream is the buffer the caller seeds with the embeddings and reads the last
// hidden states back from, one column after another, dim floats each.
func (s *Stack) Stream() *Buffer { return s.stage }

// BlockNorms is one block's seven gain vectors, in the order the block uses
// them. Naming them in a structure rather than in seven arguments is the
// difference between a call that can be read and one that cannot.
type BlockNorms struct {
	Attn        []float32 // before the attention
	PostAttn    []float32 // after it, inside the residual
	FFN         []float32 // before the shared branch
	PreFFW2     []float32 // before the expert branch
	PostFFW1    []float32 // after the shared branch
	PostFFW2    []float32 // after the expert branch
	PostFFW     []float32 // after their sum, which is what makes a mixture block
	RouterScale []float32 // the router's own, in place of a gain it has not got
	DownScale   []float32 // one scalar per expert
	Router      []float32 // the router matrix, float32, experts by dim
	OutScale    float32

	// Layout is which of the three shapes above the block's end has. Anything
	// the shape does not use is unread: a LayoutDense block reads neither the
	// router nor PreFFW2, PostFFW1 or PostFFW2, and a LayoutPreNorm block reads
	// none of those and no PostAttn or PostFFW either.
	Layout int
}

// AddBlock uploads one block's norms and router. It must be called in the
// order the blocks run, and as many times as the Attention and the Mixture
// were given blocks.
func (s *Stack) AddBlock(n BlockNorms) error {
	b := &stackBlock{outScale: n.OutScale, layout: n.Layout}
	fail := func(err error) error {
		b.close()
		return err
	}
	var err error
	// The gains a layout does not read still get a slot, so that the indices
	// the sets below are written with stay the same whatever the layout is.
	gains := [][]float32{n.Attn, n.PostAttn, n.FFN, n.PreFFW2, n.PostFFW1, n.PostFFW2, n.PostFFW}
	switch n.Layout {
	case LayoutDense:
		gains = [][]float32{n.Attn, n.PostAttn, n.FFN, n.PostFFW, n.PostFFW, n.PostFFW, n.PostFFW}
	case LayoutPreNorm:
		gains = [][]float32{n.Attn, n.Attn, n.FFN, n.FFN, n.FFN, n.FFN, n.FFN}
	}
	for _, gain := range gains {
		if len(gain) != s.dim {
			return fail(fmt.Errorf("vk: a gain should be %d wide, given %d", s.dim, len(gain)))
		}
		buf, err := s.d.Upload(asBytes(gain))
		if err != nil {
			return fail(err)
		}
		b.gains = append(b.gains, buf)
	}
	if n.Layout == LayoutMixture {
		if b.routerV, err = s.d.Upload(asBytes(n.RouterScale)); err != nil {
			return fail(err)
		}
		if b.routerW, err = s.d.Upload(asBytes(n.Router)); err != nil {
			return fail(err)
		}
		if b.downs, err = s.d.Upload(asBytes(n.DownScale)); err != nil {
			return fail(err)
		}
	}

	attnQ, attnS := s.attn.Input()
	shQ, shS := s.mix.SharedInput()
	shOut, expOut := s.mix.Outputs()
	if expOut == nil {
		// A dense mixture writes no expert branch. The binding still wants a
		// buffer, and the kernel never reads it.
		expOut = s.none
	}

	type setSpec struct {
		into **Set
		pipe *Pipeline
		bufs []*Buffer
	}
	// A pre-norm block adds the attention's answer to the stream with no norm
	// between them, so the residual reads the projection itself where the two
	// Gemma layouts read it under post_attention_norm.
	fromAttn := s.normed
	if n.Layout == LayoutPreNorm {
		fromAttn = s.attn.Output()
	}

	// a, b, gain, vscale, sum, y, yq, ys
	specs := []setSpec{
		{&b.setAttnNorm, s.norm, []*Buffer{s.xs, s.none, b.gains[0], s.none, s.none, s.none, attnQ, attnS}},
		{&b.setResid, s.norm, []*Buffer{s.xs, fromAttn, b.gains[2], s.none, s.resid, s.none, shQ, shS}},
		{&b.setCombine, s.combine, []*Buffer{shOut, expOut, s.resid, b.gains[4], b.gains[5], b.gains[6], s.xs}},
	}
	if n.Layout != LayoutPreNorm {
		specs = append(specs, setSpec{&b.setPostAttn, s.norm,
			[]*Buffer{s.attn.Output(), s.none, b.gains[1], s.none, s.none, s.normed, s.none, s.none}})
	}
	if n.Layout == LayoutMixture {
		expQ, expS := s.mix.ExpertInput()
		ids, cw := s.mix.Routing()
		specs = append(specs,
			setSpec{&b.setExpert, s.norm, []*Buffer{s.resid, s.none, b.gains[3], s.none, s.none, s.none, expQ, expS}},
			setSpec{&b.setRouterIn, s.norm, []*Buffer{s.resid, s.none, s.none, b.routerV, s.none, s.routerIn, s.none, s.none}},
			setSpec{&b.setRouterW, s.routerW, []*Buffer{s.routerIn, b.routerW, s.routerOut}},
			setSpec{&b.setPick, s.routerPick, []*Buffer{s.routerOut, b.downs, ids, cw}},
		)
	}
	for _, spec := range specs {
		if *spec.into, err = spec.pipe.NewSet(spec.bufs); err != nil {
			return fail(err)
		}
	}
	s.blocks = append(s.blocks, b)
	return nil
}

// Experts is how many logits the router produces, which Ready needs before it
// can size their buffer.
func (s *Stack) Experts(n int) error {
	if s.routerOut != nil || n == 0 {
		return nil
	}
	// shaders/router_pick.comp gives one lane to each expert.
	if n > pickLanes {
		return fmt.Errorf("vk: the router picks among at most %d experts, this model has %d", pickLanes, n)
	}
	// One row of logits per column of a pass: every position of a prompt
	// routes for itself.
	b, err := s.d.Local(n*4*maxColumns, bufferUsageStorage)
	if err != nil {
		return err
	}
	s.routerOut = b
	return nil
}

// Ready allocates the trace buffer, once every block has been added.
func (s *Stack) Ready() error {
	if s.traces != nil {
		return nil
	}
	b, err := s.d.Readback(len(s.blocks)*s.dim*4, bufferUsageStorage|bufferUsageTransferDst)
	if err != nil {
		return err
	}
	s.traces = b
	return nil
}

// Trace turns the per-block copies on. They are off by default: thirty copies
// and the thirty barriers around them are free beside a block's arithmetic and
// are not free beside the launch latency that a token is actually made of.
func (s *Stack) Trace(on bool) {
	s.trace = on
	s.forget()
}

// Profile makes the next tokens stamp the card's clock between the stages. See
// profile.go for what the stamps are worth.
func (s *Stack) Profile(t *Timeline) {
	s.tl = t
	s.forget()
	s.attn.Profile(t)
	s.mix.Profile(t)
}

// forget drops the recording, which the next Run makes again.
func (s *Stack) forget() {
	for width, p := range s.programs {
		p.Close()
		delete(s.programs, width)
	}
}

// NewTimeline is Device.NewTimeline on the device the stack was built on,
// sized for the stamps Run writes: eight a block, and a few over.
func (s *Stack) NewTimeline() (*Timeline, error) { return s.d.NewTimeline(16*len(s.blocks) + 8) }

// BlockOutput is what the given block last left in the stream. It is nil
// unless Trace was turned on before the token ran, rather than stale, so that
// a caller that forgot fails where it reads instead of comparing against the
// last thing in the buffer.
func (s *Stack) BlockOutput(block int) []float32 {
	if !s.trace {
		return nil
	}
	at := block * s.dim
	return s.traces.Floats()[at : at+s.dim]
}

// A Position says where one token goes and what each block may read. The
// window rule and the ring agree, so the range is worked out on the caller's
// side and neither kernel has to check the other.
type Position struct {
	Pos   int
	First []int // one per block
	Last  []int
}

// Run carries the stream in Stream through every block, in one submission.
//
// The recording is made on the first token and submitted again on every one
// after it. What a token changes is in the buffers: the embedding in Stream,
// the angles the card makes from it, and the position and cache range this writes
// into the attention's own. vk/compute.go says what that is worth.
func (s *Stack) Run(at []Position, experts, used int) error {
	if len(at) == 0 || len(at) > maxColumns {
		return fmt.Errorf("vk: a pass carries between one and %d columns, given %d", maxColumns, len(at))
	}
	for c, one := range at {
		if len(one.First) != len(s.blocks) || len(one.Last) != len(s.blocks) {
			return fmt.Errorf("vk: %d blocks want %d ranges, given %d", len(s.blocks), len(s.blocks), len(one.First))
		}
		for i := range s.blocks {
			if err := s.attn.SetWhere(i, c, one.Pos, one.First[i], one.Last[i]); err != nil {
				return err
			}
		}
	}
	columns := len(at)
	if s.programs == nil {
		s.programs = map[int]*Program{}
	}
	p, ok := s.programs[columns]
	if !ok {
		var err error
		if p, err = s.d.Compile(func(r *Recorder) { s.record(r, experts, used, columns) }); err != nil {
			return err
		}
		s.programs[columns] = p
	}
	return p.Run()
}

// record is the whole stack, written into a command buffer once.
func (s *Stack) record(r *Recorder, experts, used, columns int) {
	cols := uint32(columns)
	quant := normPush{n: uint32(s.dim), flags: normGain | normQuant, eps: s.eps, scalar: 1}
	post := normPush{n: uint32(s.dim), flags: normGain | normFloat, eps: s.eps, scalar: 1}
	resid := normPush{n: uint32(s.dim), flags: normAdd | normSum | normGain | normQuant, eps: s.eps, scalar: 1}
	route := routerPush{
		n: uint32(s.dim), experts: uint32(experts), used: uint32(used), columns: uint32(columns),
		eps: s.eps, scalar: float32(1 / sqrtOf(s.dim)),
	}
	// The router reads the residual normed without a gain, scaled by one over
	// the square root of the width, and multiplied elementwise by a vector the
	// file carries in place of that missing gain. gemma/moe.go says what
	// routing on anything else gives: a model that answers fluently and wrongly.
	routerIn := normPush{n: uint32(s.dim), flags: normVScale | normFloat, eps: s.eps, scalar: route.scalar}

	tl := s.tl
	if tl != nil {
		tl.Reset(r)
		tl.Stamp(r, "start")
	}
	// What the caller wrote into the stream this pass. A picture's rows reach
	// the model this way, and so does a model whose embedding is not on the
	// card; the copy is four megabytes at the widest and costs a fraction of
	// a millisecond against the eighty it saves.
	r.CopyFrom(s.xs, 0, s.stage, 0, columns*s.dim*4)
	r.Barrier()
	if s.embedSet != nil {
		// The embedding before anything, since the stream is what the first
		// block norms.
		r.Dispatch(s.embedSet, cols, unsafe.Pointer(&s.embedOf))
		r.Barrier()
		tl.Stamp(r, "embed")
	}
	// The angles first: every block reads them and nothing writes them but
	// this, so one dispatch a geometry at the head of the pass serves the
	// whole of it.
	s.attn.RecordRotations(r, columns)
	r.Barrier()
	tl.Stamp(r, "rotations")
	for i, b := range s.blocks {
		combine := combinePush{n: uint32(s.dim), eps: s.eps, outScale: b.outScale, layout: uint32(b.layout)}

		r.DispatchColumns(b.setAttnNorm, 1, cols, unsafe.Pointer(&quant))
		r.Barrier()
		tl.Stamp(r, "attn norm")
		s.attn.Record(r, i, columns)
		r.Barrier()
		tl.Stamp(r, "attn out")
		if b.layout != LayoutPreNorm {
			r.DispatchColumns(b.setPostAttn, 1, cols, unsafe.Pointer(&post))
			r.Barrier()
		}
		r.DispatchColumns(b.setResid, 1, cols, unsafe.Pointer(&resid))
		r.Barrier()
		tl.Stamp(r, "post+resid")
		sharedUp, sharedDown := false, false
		if b.layout == LayoutMixture {
			// The routing rides beside the shared branch rather than in front
			// of it. Each of these three is one workgroup, and a dispatch of
			// one workgroup costs six and a half microseconds to serialize
			// whatever is in it; beside a product that fills the card they
			// cost nothing. vk/mixture.go says what may ride with what.
			sharedUp = s.mix.RecordSharedUp(r, i, columns)
			// The expert branch's norm and the routing both read the residual
			// and neither reads the other.
			r.DispatchColumns(b.setExpert, 1, cols, unsafe.Pointer(&quant))
			r.DispatchColumns(b.setRouterIn, 1, cols, unsafe.Pointer(&routerIn))
			r.Barrier()
			tl.Stamp(r, "expert norm")
			sharedDown = s.mix.RecordSharedDown(r, i, columns)
			r.DispatchColumns(b.setRouterW, uint32(experts), uint32((columns+routerBN-1)/routerBN), unsafe.Pointer(&route))
			r.Barrier()
			tl.Stamp(r, "router")
			r.DispatchColumns(b.setPick, 1, cols, unsafe.Pointer(&route))
			r.Barrier()
			tl.Stamp(r, "pick")
		}
		s.mix.Record(r, i, columns, sharedUp, sharedDown)
		r.Barrier()
		r.DispatchColumns(b.setCombine, 1, cols, unsafe.Pointer(&combine))
		r.Barrier()
		tl.Stamp(r, "combine")
		if s.trace {
			// The last column of the pass, which is what the CPU path keeps
			// too: a batch's earlier positions are a prompt being read, and
			// the waypoint a test names is the one at its end.
			r.CopyFrom(s.traces, i*s.dim*4, s.xs, (columns-1)*s.dim*4, s.dim*4)
			r.Barrier()
		}
	}
	if s.setFinal != nil {
		// In place: the kernel loads its column into shared memory before it
		// writes anything, so the stream is both what it reads and where it
		// leaves the answer.
		final := normPush{n: uint32(s.dim), flags: normGain | normFloat, eps: s.eps, scalar: 1}
		r.DispatchColumns(s.setFinal, 1, cols, unsafe.Pointer(&final))
		r.Barrier()
		tl.Stamp(r, "final norm")
	}
	r.CopyFrom(s.stage, 0, s.xs, 0, columns*s.dim*4)
	r.Barrier()
}

// sqrtOf is math.Sqrt on an int, kept here so that this file imports no more
// than it needs.
func sqrtOf(n int) float64 {
	x := float64(n)
	if x <= 0 {
		return 1
	}
	guess := x
	for i := 0; i < 40; i++ {
		guess = 0.5 * (guess + x/guess)
	}
	return guess
}

func (s *Stack) Close() {
	if s.embedSet != nil {
		s.embedSet.Close()
		s.embedSet = nil
	}
	if s.ids != nil {
		s.ids.Close()
		s.ids = nil
	}
	if s.setFinal != nil {
		s.setFinal.Close()
		s.setFinal = nil
	}
	if s.outGain != nil {
		s.outGain.Close()
		s.outGain = nil
	}
	s.forget()
	for _, b := range s.blocks {
		b.close()
	}
	s.blocks = nil
	for _, b := range []**Buffer{&s.traces, &s.routerOut, &s.routerIn, &s.none, &s.normed, &s.resid, &s.stage, &s.xs} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	for _, p := range []**Pipeline{&s.embed, &s.combine, &s.routerPick, &s.routerW, &s.norm} {
		if *p != nil {
			(*p).Close()
			*p = nil
		}
	}
	if s.mix != nil {
		s.mix.Close()
		s.mix = nil
	}
	if s.attn != nil {
		s.attn.Close()
		s.attn = nil
	}
}

func (b *stackBlock) close() {
	for _, set := range []**Set{&b.setCombine, &b.setPick, &b.setRouterW, &b.setRouterIn, &b.setExpert, &b.setResid, &b.setPostAttn, &b.setAttnNorm} {
		if *set != nil {
			(*set).Close()
			*set = nil
		}
	}
	for i := range b.gains {
		if b.gains[i] != nil {
			b.gains[i].Close()
			b.gains[i] = nil
		}
	}
	for _, x := range []**Buffer{&b.downs, &b.routerW, &b.routerV} {
		if *x != nil {
			(*x).Close()
			*x = nil
		}
	}
}
