package vk

// Qwen3.8 out of a .golem file, on the card.
//
// Nothing about the stack changes: the same norms, the same rotation, the same
// attention, the same recurrence, the same buffers between them. What changes
// is every matrix product and what it reads.
//
// A Q4_0 product reads its activation in Q8_0 — values and scales, two buffers
// — because that is what makes it cheap. A Golem product reads floats, and it
// has to: the site's scale and the rotation are undone on this side, in float
// arithmetic, and quantizing between the two would throw away the precision
// they exist to keep. So a Golem block is the float form of every stage, with
// one transform in front of each site.
//
// There are four sites in a block and they are the same four the converter
// measured: the stream the attention norm made, which a full attention reads
// with three projections and a delta net with four; whatever the mixer
// answered; and the feed forward's two. vk/prepare_golem.go is the transform,
// nn/golem.go names the sites, and cmd/golemquant writes the vectors.

import (
	"fmt"
	"unsafe"
)

// qwenGolemFFN is one block's feed forward: three matrices and the two
// transforms their activations go through.
//
// The gate and the up projection are not stacked here, as vk/golem_ffn.go
// stacks them, because this pipeline already has a buffer for each and a swiglu
// that reads the two of them. One dispatch saved is not worth a third buffer of
// seventeen thousand floats a column.
type qwenGolemFFN struct {
	gate, up, down *GolemMatrix
	preIn, preAct  *PrepareGolem
}

// qwenGolemAttn is a full attention's four projections and its two transforms.
type qwenGolemAttn struct {
	q, k, v, o    *GolemMatrix
	preIn, preMix *PrepareGolem
}

// qwenGolemSSM is a delta net's five. Its four input projections share one site
// — they all read the stream the attention norm made — so they share one
// transform, and the output projection stands where the attention's does.
type qwenGolemSSM struct {
	qkv, gate, alpha, beta, out *GolemMatrix
	preIn, preOut               *PrepareGolem
}

// golemVectors is what a block's data carries besides its bytes: the vector of
// the site its input projections read, and the vector of the site its output
// projection reads.
type golemVectors struct {
	In  []float32
	Out []float32
}

// golem says whether this pipeline was built for a .golem checkpoint.
func (p *QwenPipeline) usesGolem() bool { return p.golem != nil }

// newGolemFFN uploads a block's feed forward and binds it to the stages this
// pipeline already has: the feed forward's norm in, its output out, and the
// gate, up and activation buffers between.
func (p *QwenPipeline) newGolemFFN(d QwenFFNData) (*qwenGolemFFN, error) {
	s := p.shape
	if len(d.PreGateUp) != s.Dim || len(d.PreDown) != s.FFN {
		return nil, fmt.Errorf("vk: a feed forward's vectors are %d and %d wide, want %d and %d",
			len(d.PreGateUp), len(d.PreDown), s.Dim, s.FFN)
	}
	b := &qwenGolemFFN{}
	var err error
	fail := func(err error) (*qwenGolemFFN, error) {
		b.Close()
		return nil, err
	}
	if b.preIn, err = p.preps.Bind(p.ffnNorm, d.PreGateUp, prepareGolemGroup); err != nil {
		return fail(err)
	}
	if b.preAct, err = p.preps.Bind(p.actBuf, d.PreDown, prepareGolemGroup); err != nil {
		return fail(err)
	}
	if b.gate, err = NewGolemMatrixOn(p.golem, d.Gate, s.FFN, s.Dim, p.ffnNorm, p.gateBuf); err != nil {
		return fail(err)
	}
	if b.up, err = NewGolemMatrixOn(p.golem, d.Up, s.FFN, s.Dim, p.ffnNorm, p.upBuf); err != nil {
		return fail(err)
	}
	if b.down, err = NewGolemMatrixOn(p.golem, d.Down, s.Dim, s.FFN, p.actBuf, p.ffnOut); err != nil {
		return fail(err)
	}
	return b, nil
}

func (p *QwenPipeline) recordGolemFFN(r *Recorder, b *qwenGolemFFN, columns int) {
	s := p.shape
	p.prepare(r, b.preIn, columns)
	r.Barrier()

	p.golemProduct(r, b.gate, columns)
	p.golemProduct(r, b.up, columns)
	r.Barrier()
	p.tl.Stamp(r, "ffn up")

	// The same gate the Q4_0 path runs. It writes the eight-bit form as well
	// and nothing here reads it, which is one pass over the activation that a
	// kernel of its own would save and is not worth a second binary.
	act := swigluPush{N: uint32(s.FFN), Columns: uint32(columns)}
	blocks := s.FFN / quantBlock * columns
	r.Dispatch(p.setAct, uint32((blocks+255)/256), unsafe.Pointer(&act))
	r.Barrier()
	p.tl.Stamp(r, "ffn act")

	p.prepare(r, b.preAct, columns)
	r.Barrier()

	p.golemProduct(r, b.down, columns)
}

func (b *qwenGolemFFN) Close() {
	for _, m := range []*GolemMatrix{b.gate, b.up, b.down} {
		if m != nil {
			m.Close()
		}
	}
	for _, x := range []*PrepareGolem{b.preIn, b.preAct} {
		if x != nil {
			x.Close()
		}
	}
	*b = qwenGolemFFN{}
}

// prepare dispatches one site's transform over the columns of a pass.
func (p *QwenPipeline) prepare(r *Recorder, x *PrepareGolem, columns int) {
	push := x.Push(columns)
	r.Dispatch(x.Set(), x.Groups(columns), unsafe.Pointer(&push))
}

// golemProduct dispatches one projection over the columns of a pass, in runs of
// the widest binary the kernels were built for.
//
// The weights are read once whatever the width, so a pass of eight costs
// barely more than a pass of one and the widest that fits is taken first. What
// is left over goes through narrower binaries rather than through a wider one
// told to answer fewer columns: a binary answers exactly the width it was
// compiled for.
func (p *QwenPipeline) golemProduct(r *Recorder, m *GolemMatrix, columns int) {
	for at := 0; at < columns; {
		w := 1
		for _, c := range GolemWidths {
			if c <= columns-at && c > w {
				w = c
			}
		}
		push := m.Push(at)
		r.Dispatch(m.Set(w), m.Groups(w), unsafe.Pointer(&push))
		at += w
	}
}

// newGolemAttn uploads a full attention's four projections and binds them to
// the stages the Q4_0 path uses, so that everything between them — the
// rotation, the caches, the softmax — is the same kernel over the same buffers.
func (p *QwenPipeline) newGolemAttn(d QwenAttnData) (*qwenGolemAttn, error) {
	s := p.shape
	if len(d.PreQKV) != s.Dim || len(d.PreO) != s.qDim() {
		return nil, fmt.Errorf("vk: an attention's vectors are %d and %d wide, want %d and %d",
			len(d.PreQKV), len(d.PreO), s.Dim, s.qDim())
	}
	b := &qwenGolemAttn{}
	var err error
	fail := func(err error) (*qwenGolemAttn, error) {
		b.Close()
		return nil, err
	}
	// The stream the norm made, transformed in place: the three projections
	// read it through one vector because they are one site.
	if b.preIn, err = p.preps.Bind(p.normed, d.PreQKV, prepareGolemGroup); err != nil {
		return fail(err)
	}
	if b.preMix, err = p.preps.Bind(p.attnOut, d.PreO, prepareGolemGroup); err != nil {
		return fail(err)
	}
	for _, u := range []struct {
		into       **GolemMatrix
		data       []byte
		rows, cols int
		in, out    *Buffer
	}{
		{&b.q, d.WQ, s.qFullDim(), s.Dim, p.normed, p.qIn},
		{&b.k, d.WK, s.kvDim(), s.Dim, p.normed, p.kIn},
		{&b.v, d.WV, s.kvDim(), s.Dim, p.normed, p.vIn},
		{&b.o, d.WO, s.Dim, s.qDim(), p.attnOut, p.mixOut},
	} {
		if *u.into, err = NewGolemMatrixOn(p.golem, u.data, u.rows, u.cols, u.in, u.out); err != nil {
			return fail(err)
		}
	}
	return b, nil
}

func (p *QwenPipeline) recordGolemAttn(r *Recorder, b *qwenGolemAttn, a *qwenAttnBlock, columns int) {
	p.prepare(r, b.preIn, columns)
	r.Barrier()

	p.golemProduct(r, b.q, columns)
	p.golemProduct(r, b.k, columns)
	p.golemProduct(r, b.v, columns)
	r.Barrier()
	p.tl.Stamp(r, "attn qkv")

	p.recordAttnMix(r, a, columns)

	p.prepare(r, b.preMix, columns)
	r.Barrier()
	p.golemProduct(r, b.o, columns)
}

func (b *qwenGolemAttn) Close() {
	for _, m := range []*GolemMatrix{b.q, b.k, b.v, b.o} {
		if m != nil {
			m.Close()
		}
	}
	for _, x := range []*PrepareGolem{b.preIn, b.preMix} {
		if x != nil {
			x.Close()
		}
	}
	*b = qwenGolemAttn{}
}

// newGolemSSM uploads a delta net's five projections. Its convolution, its
// recurrence and its two norms are the Q4_0 path's, unchanged and un-uploaded
// twice: those weights are floats in every checkpoint.
func (p *QwenPipeline) newGolemSSM(b *qwenSSMBlock, d QwenSSMData) (*qwenGolemSSM, error) {
	s := p.shape
	if len(d.PreQKV) != s.Dim || len(d.PreO) != s.Inner {
		return nil, fmt.Errorf("vk: a delta net's vectors are %d and %d wide, want %d and %d",
			len(d.PreQKV), len(d.PreO), s.Dim, s.Inner)
	}
	g := &qwenGolemSSM{}
	var err error
	fail := func(err error) (*qwenGolemSSM, error) {
		g.Close()
		return nil, err
	}
	if g.preIn, err = p.preps.Bind(p.normed, d.PreQKV, prepareGolemGroup); err != nil {
		return fail(err)
	}
	if g.preOut, err = p.preps.Bind(p.ySSM, d.PreO, prepareGolemGroup); err != nil {
		return fail(err)
	}
	for _, u := range []struct {
		into       **GolemMatrix
		data       []byte
		rows, cols int
		in, out    *Buffer
	}{
		{&g.qkv, d.WQKV, s.ConvDim, s.Dim, p.normed, p.qkvBuf},
		{&g.gate, d.WGate, s.Inner, s.Dim, p.normed, p.gateZBuf},
		{&g.alpha, d.WAlpha, s.Rank, s.Dim, p.normed, p.alphaBuf},
		{&g.beta, d.WBeta, s.Rank, s.Dim, p.normed, p.betaBuf},
		{&g.out, d.WOut, s.Dim, s.Inner, p.ySSM, p.mixOut},
	} {
		if *u.into, err = NewGolemMatrixOn(p.golem, u.data, u.rows, u.cols, u.in, u.out); err != nil {
			return fail(err)
		}
	}
	return g, nil
}

func (p *QwenPipeline) recordGolemSSM(r *Recorder, g *qwenGolemSSM, b *qwenSSMBlock, columns, snapAt int) {
	p.prepare(r, g.preIn, columns)
	r.Barrier()

	p.golemProduct(r, g.qkv, columns)
	p.golemProduct(r, g.gate, columns)
	p.golemProduct(r, g.alpha, columns)
	p.golemProduct(r, g.beta, columns)
	r.Barrier()
	p.tl.Stamp(r, "ssm in")

	p.recordSSMState(r, b, columns, snapAt)

	p.prepare(r, g.preOut, columns)
	r.Barrier()
	p.golemProduct(r, g.out, columns)
}

func (g *qwenGolemSSM) Close() {
	for _, m := range []*GolemMatrix{g.qkv, g.gate, g.alpha, g.beta, g.out} {
		if m != nil {
			m.Close()
		}
	}
	for _, x := range []*PrepareGolem{g.preIn, g.preOut} {
		if x != nil {
			x.Close()
		}
	}
	*g = qwenGolemSSM{}
}
