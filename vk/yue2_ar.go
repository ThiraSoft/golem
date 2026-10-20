package vk

// YuE2's AR half hands its keys and values to the NAR half on the card: the
// stack's cache (attention.go: a slot's ring, a position [heads][128] fp16)
// copied into the NAR's prefix cache (yue2_nar.go: the layout of
// krea2_flash.comp with GLOBAL). Twenty-eight layers of a song's prefix are
// some hundreds of megabytes, which a host round trip would carry twice.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/yue2_kvstack.comp -o shaders/yue2_kvstack.spv

//go:embed shaders/yue2_kvstack.spv
var yue2KVStackSPIRV []byte

type yue2KVStackPush struct {
	src           uint32
	heads, tokens uint32
	np            uint32
	kh, vt        uint32
}

// ImportStack copies rows first .. first + tokens - 1 of every layer's stack
// cache, ks[l] and vs[l] (heads of 128, the cache's own head count), into
// rows 0 .. tokens - 1 of the prefix cache, in one submission. first is the
// slot's offset in the ring buffers, in positions.
func (c *K2Yue2Cache) ImportStack(ks, vs []*Buffer, first, tokens int) error {
	if len(ks) != len(vs) || len(ks) == 0 {
		return fmt.Errorf("vk: %d key and %d value layers", len(ks), len(vs))
	}
	if tokens < 1 || uint32(tokens) > c.np {
		return fmt.Errorf("vk: %d keys in a YuE2 cache of %d", tokens, c.np)
	}
	if uint64(c.Keys(len(ks)-1))+2*uint64(c.heads)*128*uint64(c.np) > uint64(c.buf.Size()/2) {
		return fmt.Errorf("vk: %d layers in a YuE2 cache of fewer", len(ks))
	}
	if c.stackPipe == nil {
		p, err := c.k.d.NewPipeline(yue2KVStackSPIRV, 3, uint32(unsafe.Sizeof(yue2KVStackPush{})))
		if err != nil {
			return err
		}
		c.stackPipe = p
	}
	sets := make([]*Set, 0, len(ks))
	defer func() {
		for _, s := range sets {
			s.Close()
		}
	}()
	for l := range ks {
		s, err := c.stackPipe.NewSet([]*Buffer{c.buf, ks[l], vs[l]})
		if err != nil {
			return err
		}
		sets = append(sets, s)
	}
	return c.k.d.Submit(func(r *Recorder) {
		for l, s := range sets {
			p := yue2KVStackPush{uint32(first) * c.heads * 128, c.heads, uint32(tokens), c.np, c.Keys(l), c.Values(l)}
			r.DispatchColumns(s, uint32((tokens+63)/64), c.heads, unsafe.Pointer(&p))
		}
	})
}

// Yue2Head is YuE2's logit head on one window of its rows: Q8_0 weights as
// the file holds them, against the hidden state in float32. yue2.cpp's head
// is a mat-vec that reads the activation as it is; quantizing it to eight
// bits the way QuantHead does moves the codes' logits by 0.46% on its own
// (measured on the recording's last prefill position), which is the whole of
// the error budget a step has. shaders/matvec_q80.comp is that product.
type Yue2Head struct {
	d          *Device
	rows, cols int
	weights    *Buffer
	pipe       *Pipeline
	xs, ys     []*Buffer
	sets       []*Set
}

// NewYue2Head uploads rows rows of cols columns of Q8_0, in the file's
// blocks of 34 bytes, for up to columns hidden states a call.
func NewYue2Head(d *Device, data []byte, rows, cols, columns int) (*Yue2Head, error) {
	if cols%32 != 0 || len(data) != rows*cols/32*34 {
		return nil, fmt.Errorf("vk: %d bytes for a Q8_0 head of %d rows of %d", len(data), rows, cols)
	}
	if (rows+15)/16 > 65535 || columns < 1 {
		return nil, fmt.Errorf("vk: a head of %d rows for %d columns", rows, columns)
	}
	h := &Yue2Head{d: d, rows: rows, cols: cols}
	fail := func(err error) (*Yue2Head, error) {
		h.Close()
		return nil, err
	}
	var err error
	if h.weights, err = d.Upload(data); err != nil {
		return fail(err)
	}
	if h.pipe, err = d.NewPipeline(matvecQ80SPIRV, 3, uint32(unsafe.Sizeof(matvecKPush{}))); err != nil {
		return fail(err)
	}
	for range columns {
		x, err := d.Host(cols*4, bufferUsageStorage)
		if err != nil {
			return fail(err)
		}
		h.xs = append(h.xs, x)
		y, err := d.Readback(rows*4, bufferUsageStorage)
		if err != nil {
			return fail(err)
		}
		h.ys = append(h.ys, y)
		s, err := h.pipe.NewSet([]*Buffer{h.weights, x, y})
		if err != nil {
			return fail(err)
		}
		h.sets = append(h.sets, s)
	}
	return h, nil
}

// Logits computes outs[c] = W·xs[c] for every column, in one submission.
func (h *Yue2Head) Logits(xs, outs [][]float32) error {
	if len(xs) != len(outs) || len(xs) > len(h.sets) || len(xs) == 0 {
		return fmt.Errorf("vk: %d hidden states and %d answers for a head of %d columns", len(xs), len(outs), len(h.sets))
	}
	for c := range xs {
		if len(xs[c]) != h.cols || len(outs[c]) != h.rows {
			return fmt.Errorf("vk: a head of %d by %d given %d in and %d out", h.rows, h.cols, len(xs[c]), len(outs[c]))
		}
		copy(h.xs[c].Floats(), xs[c])
	}
	push := matvecKPush{Dim: uint32(h.rows), FFN: uint32(h.cols)}
	err := h.d.Submit(func(r *Recorder) {
		for c := range xs {
			r.Dispatch(h.sets[c], uint32((h.rows+15)/16), unsafe.Pointer(&push))
		}
	})
	if err != nil {
		return err
	}
	for c := range outs {
		copy(outs[c], h.ys[c].Floats()[:h.rows])
	}
	return nil
}

// Close gives the head back to the card.
func (h *Yue2Head) Close() {
	for _, s := range h.sets {
		s.Close()
	}
	h.sets = nil
	for _, b := range append(h.xs, h.ys...) {
		b.Close()
	}
	h.xs, h.ys = nil, nil
	if h.pipe != nil {
		h.pipe.Close()
		h.pipe = nil
	}
	if h.weights != nil {
		h.weights.Close()
		h.weights = nil
	}
}

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/yue2_attn_split.comp -o shaders/yue2_attn_split.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/yue2_attn_merge.comp -o shaders/yue2_attn_merge.spv

//go:embed shaders/yue2_attn_split.spv
var yue2AttnSplitSPIRV []byte

//go:embed shaders/yue2_attn_merge.spv
var yue2AttnMergeSPIRV []byte

// yue2Splits is how many stretches a column's visible range is cut into, and
// yue2SplitColumns how many columns of a pass may take the split attention:
// the scratch holds that many.
const (
	yue2Splits       = 16
	yue2SplitColumns = 8
)

type yue2SplitPush struct {
	scorePush
	splits uint32
}

// yue2Split is the split attention of the runs of one column, which an
// Attention takes only when UseSplitDecode asks for it.
type yue2Split struct {
	split, merge *Pipeline
	part         *Buffer
	splitSets    []*Set // by block
	mergeSet     *Set
}

// UseSplitDecode makes every run of a pass within its first
// yue2SplitColumns columns (a token being drawn, a guided pair, a stretch of
// a prompt read eight at a time) read its keys a column at a time through
// yue2_attn_split.comp and yue2_attn_merge.comp, rather than through
// attn_one.comp or attn_scores.comp: sixteen workgroups to a head and a
// column instead of one to a head. It is called once every block has been
// added; an Attention that never calls it records what it always did.
func (a *Attention) UseSplitDecode() error {
	if a.split != nil {
		return nil
	}
	if a.scoresFloat != nil {
		return fmt.Errorf("vk: the split attention writes the Q8_0 form only")
	}
	maxD := 0
	for _, b := range a.blocks {
		if b.golem != nil {
			return fmt.Errorf("vk: the split attention writes the Q8_0 form only")
		}
		maxD = max(maxD, b.shape.HeadDim)
	}
	s := &yue2Split{}
	fail := func(err error) error {
		s.close()
		return err
	}
	var err error
	if s.split, err = a.d.NewPipeline(yue2AttnSplitSPIRV, 5, uint32(unsafe.Sizeof(yue2SplitPush{}))); err != nil {
		return fail(err)
	}
	if s.merge, err = a.d.NewPipeline(yue2AttnMergeSPIRV, 3, uint32(unsafe.Sizeof(yue2SplitPush{}))); err != nil {
		return fail(err)
	}
	maxHeads := 0
	for _, b := range a.blocks {
		maxHeads = max(maxHeads, b.shape.Heads)
	}
	if s.part, err = a.d.Local(yue2SplitColumns*maxHeads*yue2Splits*(maxD+2)*4, bufferUsageStorage); err != nil {
		return fail(err)
	}
	for _, b := range a.blocks {
		ck, cv := b.caches()
		if ck == nil {
			ck, cv = b.shape.Cache[0], b.shape.Cache[1]
		}
		set, err := s.split.NewSet([]*Buffer{a.qh, ck, cv, s.part, a.where})
		if err != nil {
			return fail(err)
		}
		s.splitSets = append(s.splitSets, set)
	}
	if s.mergeSet, err = s.merge.NewSet([]*Buffer{s.part, a.aq, a.as}); err != nil {
		return fail(err)
	}
	a.split = s
	return nil
}

// recordSplitRun records the split attention of every column of a run, the
// stretches of all of them then their folds, and says whether it did: a run
// reaching past column yue2SplitColumns is left to the kernels it always
// took.
func (a *Attention) recordSplitRun(r *Recorder, block int, run span, score scorePush) bool {
	s := a.split
	if run.first+run.count > yue2SplitColumns {
		return false
	}
	push := yue2SplitPush{score, yue2Splits}
	for c := 0; c < run.count; c++ {
		push.col0 = uint32(run.first + c)
		push.columns = push.col0 + 1
		r.DispatchColumns(s.splitSets[block], score.heads, yue2Splits, unsafe.Pointer(&push))
	}
	r.Barrier()
	for c := 0; c < run.count; c++ {
		push.col0 = uint32(run.first + c)
		push.columns = push.col0 + 1
		r.Dispatch(s.mergeSet, score.heads, unsafe.Pointer(&push))
	}
	return true
}

func (s *yue2Split) close() {
	if s == nil {
		return
	}
	for _, set := range append(s.splitSets, s.mergeSet) {
		if set != nil {
			set.Close()
		}
	}
	s.splitSets, s.mergeSet = nil, nil
	if s.part != nil {
		s.part.Close()
		s.part = nil
	}
	for _, p := range []**Pipeline{&s.split, &s.merge} {
		if *p != nil {
			(*p).Close()
			*p = nil
		}
	}
}
