package vk

// The four projections of an attention, on the card.
//
// What is here is only the products: the queries, the keys, the values and the
// output projection. The norms, the rotation, the cache and the scores stay
// where they were. That is not where the bytes are — a block's four matrices
// are nineteen megabytes and the scores are a few kilobytes — and it is where
// all of Gemma 4's particulars live: a query norm and a key norm, two rotation
// geometries, a value that is sometimes the key before the key was rotated,
// and fifteen blocks at the end that compute no keys at all and read what two
// earlier blocks left behind.
//
// So this moves 487 megabytes a token and leaves the intricacy alone. What it
// costs is two more submissions a block, one before the attention and one
// after, because the attention between them is on the other side.
//
// The kernel is shaders/matvec.comp, which is the mixture's own down
// projection with the mixture taken out.

import (
	"fmt"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// An Attention is every attention matrix of a model, resident, and the buffers
// one position passes through.
type Attention struct {
	d *Device

	dim             int // the stream's width, which is what the output projection makes
	maxHeads, maxKV int // the widest block's query and key projections, for the buffers

	matvec *Pipeline

	// The input to the three projections, and their outputs.
	xq, xs      *Buffer
	oq, ok_, ov *Buffer
	// The output projection's input and output.
	aq, as, out *Buffer

	blocks []*attentionBlock
}

// An attentionBlock is one block's four matrices. K and V are absent on the
// blocks that compute neither.
type attentionBlock struct {
	q, k, v, o *Buffer
	setQ, setK *Set
	setV, setO *Set

	// This block's own widths. Two geometries alternate through Gemma 4 and
	// their heads are not the same size: sixteen of 256 on the blocks that see
	// a window, sixteen of 512 on the ones that see everything.
	heads, kv int
}

// NewAttention builds the kernel and the shared buffers, which are sized for
// the widest block: maxHeads is the largest query projection's output width
// and maxKV the largest key's, both counted in floats.
func NewAttention(d *Device, dim, maxHeads, maxKV int) (*Attention, error) {
	for _, n := range []int{dim, maxHeads, maxKV} {
		if n%nn.QuantBlock != 0 {
			return nil, fmt.Errorf("vk: attention shapes must be multiples of %d, given %d", nn.QuantBlock, n)
		}
	}
	a := &Attention{d: d, dim: dim, maxHeads: maxHeads, maxKV: maxKV}

	var err error
	if a.matvec, err = d.NewPipeline(matvecSPIRV, 4, uint32(unsafe.Sizeof(moePush{}))); err != nil {
		return nil, err
	}
	for _, spec := range []struct {
		into **Buffer
		size int
	}{
		{&a.xq, dim}, // the normed stream, Q8_0
		{&a.xs, 2 * dim / nn.QuantBlock * 4},
		{&a.oq, maxHeads * 4}, // the queries
		{&a.ok_, maxKV * 4},   // the keys
		{&a.ov, maxKV * 4},    // the values
		{&a.aq, maxHeads},     // the attention's own output, Q8_0
		{&a.as, 2 * maxHeads / nn.QuantBlock * 4},
		{&a.out, dim * 4}, // and what the output projection makes of it
	} {
		if *spec.into, err = d.Host(spec.size, bufferUsageStorage); err != nil {
			a.Close()
			return nil, err
		}
	}
	return a, nil
}

// Blocks is how many have been added.
func (a *Attention) Blocks() int { return len(a.blocks) }

// AddBlock uploads one block's matrices in the file's own layout. k and v are
// nil on a block that computes neither and reads an earlier block's cache, and
// v alone is nil where the value is the key before the key was rotated.
func (a *Attention) AddBlock(q, k, v, o []byte, heads, kv int) error {
	if k == nil && v != nil {
		return fmt.Errorf("vk: a block with values and no keys")
	}
	if heads > a.maxHeads || kv > a.maxKV {
		return fmt.Errorf("vk: block %d attends over %d and %d, past the %d and %d the buffers hold",
			len(a.blocks), heads, kv, a.maxHeads, a.maxKV)
	}
	if heads%nn.QuantBlock != 0 || kv%nn.QuantBlock != 0 {
		return fmt.Errorf("vk: attention shapes must be multiples of %d, given %d and %d", nn.QuantBlock, heads, kv)
	}
	for _, spec := range []struct {
		what       string
		data       []byte
		rows, cols int
	}{
		{"the query projection", q, heads, a.dim},
		{"the key projection", k, kv, a.dim},
		{"the value projection", v, kv, a.dim},
		{"the output projection", o, a.dim, heads},
	} {
		if spec.data == nil {
			continue
		}
		if want := spec.rows * rowBytesQ4_0(spec.cols); len(spec.data) != want {
			return fmt.Errorf("vk: %s should be %d bytes, given %d", spec.what, want, len(spec.data))
		}
	}

	b := &attentionBlock{heads: heads, kv: kv}
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
		{&b.q, &b.setQ, q, heads, a.dim, a.oq},
		{&b.k, &b.setK, k, kv, a.dim, a.ok_},
		{&b.v, &b.setV, v, kv, a.dim, a.ov},
		{&b.o, &b.setO, o, a.dim, heads, a.out},
	} {
		if spec.data == nil {
			continue
		}
		if *spec.into, err = a.d.Upload(splitQ4_0(spec.data, spec.rows, spec.cols)); err != nil {
			return fail(err)
		}
		in, scales := a.xq, a.xs
		if spec.out == a.out {
			in, scales = a.aq, a.as // the output projection reads the attention, not the stream
		}
		if *spec.set, err = a.matvec.NewSet([]*Buffer{*spec.into, in, scales, spec.out}); err != nil {
			return fail(err)
		}
	}
	a.blocks = append(a.blocks, b)
	return nil
}

// QKV computes the three projections of one position in one submission. k and
// v are written only when the block has those matrices; a caller whose block
// has none passes nil for both and gets the queries alone.
func (a *Attention) QKV(block int, in *nn.Batch, q, k, v []float32) error {
	b, err := a.at(block)
	if err != nil {
		return err
	}
	if err := a.load(in, a.xq, a.xs, a.dim); err != nil {
		return err
	}
	for _, spec := range []struct {
		want []float32
		set  *Set
		what string
	}{{k, b.setK, "keys"}, {v, b.setV, "values"}} {
		if spec.want != nil && spec.set == nil {
			return fmt.Errorf("vk: block %d has no %s to compute", block, spec.what)
		}
	}
	push := moePush{dim: uint32(b.heads), ffn: uint32(a.dim), used: 1}
	kvPush := moePush{dim: uint32(b.kv), ffn: uint32(a.dim), used: 1}
	err = a.d.Submit(func(r *Recorder) {
		// The three read the same input and none of them reads another, so
		// they go in together and the card runs them at once.
		r.Dispatch(b.setQ, groups(b.heads), unsafe.Pointer(&push))
		if k != nil {
			r.Dispatch(b.setK, groups(b.kv), unsafe.Pointer(&kvPush))
		}
		if v != nil {
			r.Dispatch(b.setV, groups(b.kv), unsafe.Pointer(&kvPush))
		}
	})
	if err != nil {
		return err
	}
	copy(q, a.oq.Floats()[:b.heads])
	if k != nil {
		copy(k, a.ok_.Floats()[:b.kv])
	}
	if v != nil {
		copy(v, a.ov.Floats()[:b.kv])
	}
	return nil
}

// Out is the output projection, which reads what the attention made and not
// the stream.
func (a *Attention) Out(block int, in *nn.Batch, out []float32) error {
	b, err := a.at(block)
	if err != nil {
		return err
	}
	if err := a.load(in, a.aq, a.as, b.heads); err != nil {
		return err
	}
	if len(out) != a.dim {
		return fmt.Errorf("vk: the output projection writes %d, given %d", a.dim, len(out))
	}
	push := moePush{dim: uint32(a.dim), ffn: uint32(b.heads), used: 1}
	if err := b.setO.Dispatch(groups(a.dim), unsafe.Pointer(&push)); err != nil {
		return err
	}
	copy(out, a.out.Floats()[:a.dim])
	return nil
}

// load copies one column's Q8_0 form into the buffers a kernel reads.
func (a *Attention) load(in *nn.Batch, q, s *Buffer, width int) error {
	if in.Width != width || in.Size != 1 {
		return fmt.Errorf("vk: a projection reads one column of %d, given %d of %d", width, in.Size, in.Width)
	}
	if in.Q == nil {
		return fmt.Errorf("vk: a projection needs its input in the Q8_0 form")
	}
	dst := q.Bytes()
	for i, value := range in.Q[:width] {
		dst[i] = byte(value)
	}
	blocks := width / nn.QuantBlock
	scales := s.Floats()
	copy(scales[:blocks], in.Scales[:blocks])
	copy(scales[blocks:2*blocks], in.Corr[:blocks])
	return nil
}

func (a *Attention) at(block int) (*attentionBlock, error) {
	if block < 0 || block >= len(a.blocks) {
		return nil, fmt.Errorf("vk: block %d of %d", block, len(a.blocks))
	}
	return a.blocks[block], nil
}

// groups is how many workgroups the matvec kernel needs for that many outputs.
func groups(outputs int) uint32 { return uint32((outputs + 63) / 64) }

func (a *Attention) Close() {
	for _, b := range a.blocks {
		b.close()
	}
	a.blocks = nil
	for _, b := range []**Buffer{&a.out, &a.as, &a.aq, &a.ov, &a.ok_, &a.oq, &a.xs, &a.xq} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	if a.matvec != nil {
		a.matvec.Close()
		a.matvec = nil
	}
}

func (b *attentionBlock) close() {
	for _, s := range []**Set{&b.setO, &b.setV, &b.setK, &b.setQ} {
		if *s != nil {
			(*s).Close()
			*s = nil
		}
	}
	for _, x := range []**Buffer{&b.o, &b.v, &b.k, &b.q} {
		if *x != nil {
			(*x).Close()
			*x = nil
		}
	}
}
