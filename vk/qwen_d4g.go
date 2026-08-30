package vk

// Qwen3.8 out of a .golem file, on the card.
//
// Nothing about the stack changes: the same norms, the same rotation, the same
// attention, the same recurrence, the same buffers between them. What changes
// is every matrix product and what it reads.
//
// A Q4_0 product reads its activation in Q8_0 — values and scales, two buffers
// — because that is what makes it cheap. A D4G product reads floats, and it has
// to: the site's scale and the rotation are undone on this side, in float
// arithmetic, and quantizing between the two would throw away the precision
// they exist to keep. So a D4G block is the float form of every stage, with
// one transform in front of each site.
//
// There are four sites in a block and they are the same four the converter
// measured: the stream the attention norm made, which a full attention reads
// with three projections and a delta net with four; whatever the mixer
// answered; and the feed forward's two. vk/prepare_d4g.go is the transform,
// nn/d4g.go names the sites, and cmd/golemquant writes the vectors.

import (
	"fmt"
	"unsafe"
)

// qwenD4GFFN is one block's feed forward: three matrices and the two transforms
// their activations go through.
//
// The gate and the up projection are not stacked here, as vk/d4g_ffn.go stacks
// them, because this pipeline already has a buffer for each and a swiglu that
// reads the two of them. One dispatch saved is not worth a third buffer of
// seventeen thousand floats a column.
type qwenD4GFFN struct {
	gate, up, down *D4GMatrix
	preIn, preAct  *PrepareD4G
}

// qwenD4GAttn is a full attention's four projections and its two transforms.
type qwenD4GAttn struct {
	q, k, v, o    *D4GMatrix
	preIn, preMix *PrepareD4G
}

// qwenD4GSSM is a delta net's five. Its four input projections share one site
// — they all read the stream the attention norm made — so they share one
// transform, and the output projection stands where the attention's does.
type qwenD4GSSM struct {
	qkv, gate, alpha, beta, out *D4GMatrix
	preIn, preOut               *PrepareD4G
}

// d4gVectors is what a block's data carries besides its bytes: the vector of
// the site its input projections read, and the vector of the site its output
// projection reads.
type d4gVectors struct {
	In  []float32
	Out []float32
}

// d4g says whether this pipeline was built for a .golem checkpoint.
func (p *QwenPipeline) usesD4G() bool { return p.d4g != nil }

// newD4GFFN uploads a block's feed forward and binds it to the stages this
// pipeline already has: the feed forward's norm in, its output out, and the
// gate, up and activation buffers between.
func (p *QwenPipeline) newD4GFFN(d QwenFFNData) (*qwenD4GFFN, error) {
	s := p.shape
	if len(d.PreGateUp) != s.Dim || len(d.PreDown) != s.FFN {
		return nil, fmt.Errorf("vk: a feed forward's vectors are %d and %d wide, want %d and %d",
			len(d.PreGateUp), len(d.PreDown), s.Dim, s.FFN)
	}
	b := &qwenD4GFFN{}
	var err error
	fail := func(err error) (*qwenD4GFFN, error) {
		b.Close()
		return nil, err
	}
	if b.preIn, err = p.preps.Bind(p.ffnNorm, d.PreGateUp, prepareD4GGroup); err != nil {
		return fail(err)
	}
	if b.preAct, err = p.preps.Bind(p.actBuf, d.PreDown, prepareD4GGroup); err != nil {
		return fail(err)
	}
	if b.gate, err = NewD4GMatrixOn(p.d4g, d.Gate, s.FFN, s.Dim, p.ffnNorm, p.gateBuf); err != nil {
		return fail(err)
	}
	if b.up, err = NewD4GMatrixOn(p.d4g, d.Up, s.FFN, s.Dim, p.ffnNorm, p.upBuf); err != nil {
		return fail(err)
	}
	if b.down, err = NewD4GMatrixOn(p.d4g, d.Down, s.Dim, s.FFN, p.actBuf, p.ffnOut); err != nil {
		return fail(err)
	}
	return b, nil
}

func (p *QwenPipeline) recordD4GFFN(r *Recorder, b *qwenD4GFFN, columns int) {
	s := p.shape
	p.prepare(r, b.preIn, columns)
	r.Barrier()

	p.d4gProduct(r, b.gate, columns)
	p.d4gProduct(r, b.up, columns)
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

	p.d4gProduct(r, b.down, columns)
}

func (b *qwenD4GFFN) Close() {
	for _, m := range []*D4GMatrix{b.gate, b.up, b.down} {
		if m != nil {
			m.Close()
		}
	}
	for _, x := range []*PrepareD4G{b.preIn, b.preAct} {
		if x != nil {
			x.Close()
		}
	}
	*b = qwenD4GFFN{}
}

// prepare dispatches one site's transform over the columns of a pass.
func (p *QwenPipeline) prepare(r *Recorder, x *PrepareD4G, columns int) {
	push := x.Push(columns)
	r.Dispatch(x.Set(), x.Groups(columns), unsafe.Pointer(&push))
}

// d4gProduct dispatches one projection over the columns of a pass, in runs of
// the widest binary the kernels were built for.
//
// The weights are read once whatever the width, so a pass of eight costs
// barely more than a pass of one and the widest that fits is taken first. What
// is left over goes through narrower binaries rather than through a wider one
// told to answer fewer columns: a binary answers exactly the width it was
// compiled for.
func (p *QwenPipeline) d4gProduct(r *Recorder, m *D4GMatrix, columns int) {
	for at := 0; at < columns; {
		w := 1
		for _, c := range D4GWidths {
			if c <= columns-at && c > w {
				w = c
			}
		}
		push := m.Push(at)
		r.Dispatch(m.Set(w), m.Groups(), unsafe.Pointer(&push))
		at += w
	}
}

// newD4GAttn uploads a full attention's four projections and binds them to the
// stages the Q4_0 path uses, so that everything between them — the rotation,
// the caches, the softmax — is the same kernel over the same buffers.
func (p *QwenPipeline) newD4GAttn(d QwenAttnData) (*qwenD4GAttn, error) {
	s := p.shape
	if len(d.PreQKV) != s.Dim || len(d.PreO) != s.qDim() {
		return nil, fmt.Errorf("vk: an attention's vectors are %d and %d wide, want %d and %d",
			len(d.PreQKV), len(d.PreO), s.Dim, s.qDim())
	}
	b := &qwenD4GAttn{}
	var err error
	fail := func(err error) (*qwenD4GAttn, error) {
		b.Close()
		return nil, err
	}
	// The stream the norm made, transformed in place: the three projections
	// read it through one vector because they are one site.
	if b.preIn, err = p.preps.Bind(p.normed, d.PreQKV, prepareD4GGroup); err != nil {
		return fail(err)
	}
	if b.preMix, err = p.preps.Bind(p.attnOut, d.PreO, prepareD4GGroup); err != nil {
		return fail(err)
	}
	for _, u := range []struct {
		into       **D4GMatrix
		data       []byte
		rows, cols int
		in, out    *Buffer
	}{
		{&b.q, d.WQ, s.qFullDim(), s.Dim, p.normed, p.qIn},
		{&b.k, d.WK, s.kvDim(), s.Dim, p.normed, p.kIn},
		{&b.v, d.WV, s.kvDim(), s.Dim, p.normed, p.vIn},
		{&b.o, d.WO, s.Dim, s.qDim(), p.attnOut, p.mixOut},
	} {
		if *u.into, err = NewD4GMatrixOn(p.d4g, u.data, u.rows, u.cols, u.in, u.out); err != nil {
			return fail(err)
		}
	}
	return b, nil
}

func (p *QwenPipeline) recordD4GAttn(r *Recorder, b *qwenD4GAttn, a *qwenAttnBlock, columns int) {
	p.prepare(r, b.preIn, columns)
	r.Barrier()

	p.d4gProduct(r, b.q, columns)
	p.d4gProduct(r, b.k, columns)
	p.d4gProduct(r, b.v, columns)
	r.Barrier()
	p.tl.Stamp(r, "attn qkv")

	p.recordAttnMix(r, a, columns)

	p.prepare(r, b.preMix, columns)
	r.Barrier()
	p.d4gProduct(r, b.o, columns)
}

func (b *qwenD4GAttn) Close() {
	for _, m := range []*D4GMatrix{b.q, b.k, b.v, b.o} {
		if m != nil {
			m.Close()
		}
	}
	for _, x := range []*PrepareD4G{b.preIn, b.preMix} {
		if x != nil {
			x.Close()
		}
	}
	*b = qwenD4GAttn{}
}

// newD4GSSM uploads a delta net's five projections. Its convolution, its
// recurrence and its two norms are the Q4_0 path's, unchanged and un-uploaded
// twice: those weights are floats in every checkpoint.
func (p *QwenPipeline) newD4GSSM(b *qwenSSMBlock, d QwenSSMData) (*qwenD4GSSM, error) {
	s := p.shape
	if len(d.PreQKV) != s.Dim || len(d.PreO) != s.Inner {
		return nil, fmt.Errorf("vk: a delta net's vectors are %d and %d wide, want %d and %d",
			len(d.PreQKV), len(d.PreO), s.Dim, s.Inner)
	}
	g := &qwenD4GSSM{}
	var err error
	fail := func(err error) (*qwenD4GSSM, error) {
		g.Close()
		return nil, err
	}
	if g.preIn, err = p.preps.Bind(p.normed, d.PreQKV, prepareD4GGroup); err != nil {
		return fail(err)
	}
	if g.preOut, err = p.preps.Bind(p.ySSM, d.PreO, prepareD4GGroup); err != nil {
		return fail(err)
	}
	for _, u := range []struct {
		into       **D4GMatrix
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
		if *u.into, err = NewD4GMatrixOn(p.d4g, u.data, u.rows, u.cols, u.in, u.out); err != nil {
			return fail(err)
		}
	}
	return g, nil
}

func (p *QwenPipeline) recordD4GSSM(r *Recorder, g *qwenD4GSSM, b *qwenSSMBlock, columns, snapAt int) {
	p.prepare(r, g.preIn, columns)
	r.Barrier()

	p.d4gProduct(r, g.qkv, columns)
	p.d4gProduct(r, g.gate, columns)
	p.d4gProduct(r, g.alpha, columns)
	p.d4gProduct(r, g.beta, columns)
	r.Barrier()
	p.tl.Stamp(r, "ssm in")

	p.recordSSMState(r, b, columns, snapAt)

	p.prepare(r, g.preOut, columns)
	r.Barrier()
	p.d4gProduct(r, g.out, columns)
}

func (g *qwenD4GSSM) Close() {
	for _, m := range []*D4GMatrix{g.qkv, g.gate, g.alpha, g.beta, g.out} {
		if m != nil {
			m.Close()
		}
	}
	for _, x := range []*PrepareD4G{g.preIn, g.preOut} {
		if x != nil {
			x.Close()
		}
	}
	*g = qwenD4GSSM{}
}
