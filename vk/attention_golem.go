package vk

// The four projections of an attention, in the Golem format.
//
// Everything between them is unchanged: the query and key norms, the rotation,
// the cache, the scores and the mix are the same kernels reading the same
// buffers, because none of them touches a weight. What changes is only how the
// four matrices are read, and what their operand is.
//
// The operand is the reason this is a mode rather than a format flag. A Q4_0
// projection reads its activation in Q8_0 — values and scales, two bindings —
// and a Golem one reads floats, because the scale and the rotation this format
// undoes on the activation side are float arithmetic. So the block's input has
// to arrive as floats, which is what FloatInput is for, and the mix has to be
// turned back into floats before the output projection, which the prepare
// kernel does on its way through rather than in a pass of its own.

import (
	"fmt"
	"unsafe"
)

// golemAttn is one block's Golem side: the two transforms and the four
// matrices.
type golemAttn struct {
	prepIn  *PrepareGolem // the normed stream, in place
	prepMix *PrepareGolem // the mix, from its Q8_0 form into af
	q, k, v *GolemMatrix
	o       *GolemMatrix
}

func (x *golemAttn) close() {
	if x == nil {
		return
	}
	for _, p := range []*PrepareGolem{x.prepIn, x.prepMix} {
		if p != nil {
			p.Close()
		}
	}
	for _, m := range []*GolemMatrix{x.q, x.k, x.v, x.o} {
		if m != nil {
			m.Close()
		}
	}
}

// FloatInput is where a Golem attention reads its normed stream. The stack
// binds the norm's float output to it, in place of the Q8_0 pair Input returns,
// and the two are never both written.
func (a *Attention) FloatInput() *Buffer { return a.xf }

// Golem says whether this attention's blocks read Golem weights.
func (a *Attention) Golem() bool { return len(a.blocks) > 0 && a.blocks[0].golem != nil }

// AddBlockGolem is AddBlock for a block whose four projections are Golem.
// preQKV is the vector the stream meets before the three input projections and
// preO the one the mix meets before the output projection — one per site, which
// is what the file carries.
func (a *Attention) AddBlockGolem(k *GolemKernels, shape BlockShape,
	q, kw, v, o []byte, qnorm, knorm, preQKV, preO []float32) error {
	heads := shape.Heads * shape.HeadDim
	kv := shape.KVHeads * shape.HeadDim
	if len(preQKV) != a.dim {
		return fmt.Errorf("vk: the qkv vector is %d wide, the stream is %d", len(preQKV), a.dim)
	}
	if len(preO) != heads {
		return fmt.Errorf("vk: the output vector is %d wide, the mix is %d", len(preO), heads)
	}
	if err := a.ensureGolemBuffers(); err != nil {
		return err
	}
	if a.scoresFloat == nil {
		p, err := a.d.NewPipeline(scoresFloatSPIRV, 8, uint32(unsafe.Sizeof(scorePush{})))
		if err != nil {
			return err
		}
		a.scoresFloat = p
	}
	// The block itself, without its projections: AddBlock does the norms, the
	// cache and the sets that read them, and skips a projection whose bytes
	// are nil.
	if err := a.AddBlock(shape, nil, nil, nil, nil, qnorm, knorm); err != nil {
		return err
	}
	b := a.blocks[len(a.blocks)-1]
	x := &golemAttn{}
	fail := func(err error) error {
		x.close()
		return err
	}
	var err error
	if x.prepIn, err = NewPrepareGolem(a.d, a.xf, preQKV, prepareGolemGroup); err != nil {
		return fail(err)
	}
	// The mix arrives as floats, because the scores kernel this attention was
	// built with writes them beside its Q8_0 form. Reading the quantized one
	// instead would cost a fifth of a percent at every block and compound
	// through the depth of the model: measured, 0.19% at block 0 and 2.4% by
	// the end, which is a different answer and not a rounding of the same one.
	if x.prepMix, err = NewPrepareGolem(a.d, a.af, preO, prepareGolemGroup); err != nil {
		return fail(err)
	}
	for _, spec := range []struct {
		into       **GolemMatrix
		data       []byte
		rows, cols int
		in, out    *Buffer
	}{
		{&x.q, q, heads, a.dim, a.xf, a.q},
		{&x.k, kw, kv, a.dim, a.xf, a.k},
		{&x.v, v, kv, a.dim, a.xf, a.v},
		{&x.o, o, a.dim, heads, a.af, a.out},
	} {
		if spec.data == nil {
			continue
		}
		if *spec.into, err = NewGolemMatrixOn(k, spec.data, spec.rows, spec.cols, spec.in, spec.out); err != nil {
			return fail(err)
		}
	}
	b.golem = x
	return nil
}

// ensureGolemBuffers allocates the two the Golem path needs and the Q8_0 one
// does not: the normed stream as floats, and the mix as floats.
func (a *Attention) ensureGolemBuffers() error {
	if a.xf != nil {
		return nil
	}
	var err error
	if a.xf, err = a.d.Local(a.dim*4*maxColumns, bufferUsageStorage); err != nil {
		return err
	}
	if a.af, err = a.d.Local(a.maxHeads*4*maxColumns, bufferUsageStorage); err != nil {
		return err
	}
	return nil
}

// recordGolemInput is the three input projections: the stream through the
// site's vector and rotation, then Q, K and V.
func (a *Attention) recordGolemInput(r *Recorder, b *attentionBlock, columns int) {
	x := b.golem
	push := x.prepIn.Push(columns)
	r.Dispatch(x.prepIn.Set(), x.prepIn.Groups(columns), unsafe.Pointer(&push))
	r.Barrier()
	width, _ := golemPassWidth(columns)
	for _, m := range []*GolemMatrix{x.q, x.k, x.v} {
		if m == nil {
			continue
		}
		for first := 0; first < columns; first += width {
			p := m.Push(first)
			r.Dispatch(m.Set(width), m.Groups(), unsafe.Pointer(&p))
		}
	}
}

// recordGolemOutput is the output projection: the mix out of its Q8_0 form,
// through its own vector and rotation, and into the stream.
func (a *Attention) recordGolemOutput(r *Recorder, b *attentionBlock, columns int) {
	x := b.golem
	push := x.prepMix.Push(columns)
	r.Dispatch(x.prepMix.Set(), x.prepMix.Groups(columns), unsafe.Pointer(&push))
	r.Barrier()
	width, _ := golemPassWidth(columns)
	for first := 0; first < columns; first += width {
		p := x.o.Push(first)
		r.Dispatch(x.o.Set(width), x.o.Groups(), unsafe.Pointer(&p))
	}
}
