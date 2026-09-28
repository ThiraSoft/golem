package vk

// Laya on the card: ModernBERT-large and the decision head above it.
//
// It is nomic's kind of work (every position at once, several sequences in a
// pass, fp16 weights) and it started on nomic's kernels. What differs is the
// block. ModernBERT is pre-norm and has no biases; its feed forward is a gated
// GELU on the error function; two blocks in three see only sixty-four
// positions either side, which the attention takes as a window; and each
// block turns its queries by one of two bases, from a table the host computes
// as PyTorch does.
//
// And what differs most is the precision. ModernBERT grows activations in the
// thousands at a few positions from its nineteenth block, and whether a
// position does is a threshold: with every product's operand rounded to fp16,
// one position of the long fixture crossed it and ended half a unit away from
// PyTorch. So every operand is carried as two fp16 planes, its high part and
// what is left of it, and both meet the weights on the matrix cores, which
// makes the product the float32 operand's to about twenty-two bits for twice
// the work. The kernels that write an operand (the norm, the attention, the
// gated GELU) write it split, once, and shaders/laya_mm.comp reads the planes
// as they are. Its epilogue does the rest of a block's elementwise work on the
// way out: the residual add, the gated GELU (the up projection's two halves
// are interleaved row by row at upload so that a tile holds both), the ReLU
// and the biases of the head.
//
// The head is two of PyTorch's TransformerEncoderLayer, pre-norm, with biases
// and a ReLU, after each position has had its question's kind added. It runs
// here too, and what comes back is every position after it: the host reads
// the markers and scores them, which is a few rows.

import (
	_ "embed"
	"fmt"
	"math"
	"time"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/laya_act.comp -o shaders/laya_act.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/laya_rows.comp -o shaders/laya_rows.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/laya_norm.comp -o shaders/laya_norm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/laya_attn.comp -o shaders/laya_attn.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/laya_reduce.comp -o shaders/laya_reduce.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DLANES=32 -DCHUNK=256 shaders/laya_mm.comp -o shaders/laya_mm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DLANES=32 -DBM=64 -DBN=64 -DCHUNK=256 shaders/laya_mm.comp -o shaders/laya_mm_small.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/laya_mm_scalar.comp -o shaders/laya_mm_scalar.spv

//go:embed shaders/laya_act.spv
var layaActSPIRV []byte

//go:embed shaders/laya_rows.spv
var layaRowsSPIRV []byte

//go:embed shaders/laya_norm.spv
var layaNormSPIRV []byte

//go:embed shaders/laya_attn.spv
var layaAttnSPIRV []byte

//go:embed shaders/laya_reduce.spv
var layaReduceSPIRV []byte

//go:embed shaders/laya_mm.spv
var layaMMSPIRV []byte

//go:embed shaders/laya_mm_small.spv
var layaMMSmallSPIRV []byte

//go:embed shaders/laya_mm_scalar.spv
var layaMMScalarSPIRV []byte

// The wave the products are compiled for: shaders/laya_mm.comp's LANES.
const layaWave = 32

// The tiles of shaders/laya_mm.comp: the large one, BM by BN, and the small.
const (
	layaBig   = 128
	laySmall  = 64
	layaSlice = 256 // the least of the shared dimension a slice of a split product walks
)

// layaQueryTile is how many queries of one head a unit of
// shaders/laya_attn.comp takes: its BR.
const layaQueryTile = 32

// The operations of shaders/laya_act.comp.
const (
	layaSplit = 0
	layaKind  = 1
)

// The operations of shaders/laya_rows.comp.
const (
	layaEmbed  = 0
	layaGather = 1
)

// The epilogues of shaders/laya_epi.glsl.
const (
	layaStore = 0
	layaAdd   = 1
	layaGeGLU = 2
	layaReLU  = 3
	layaGELU  = 4
)

// LayaShape is the model's geometry.
type LayaShape struct {
	Dim, Heads, HeadDim int
	FF                  int // the encoder's, half of what its up projection writes
	HeadFF              int
	Window              int // how far a local block sees either side
	MaxLen              int // the longest sequence, which sizes the rotation tables
	Eps                 float32
}

// LayaEncoderBlock is one ModernBERT block. AttnGain is nil for the first,
// which has no norm before its attention.
type LayaEncoderBlock struct {
	AttnGain, MLPGain []float32
	QKV, O, Up, Down  NomicLinear
	Global            bool
}

// LayaHeadBlock is one layer of the head.
type LayaHeadBlock struct {
	Norm1Gain, Norm1Bias, Norm2Gain, Norm2Bias []float32
	QKV, O, Up, Down                           NomicLinear
}

type layaEncoderAt struct {
	attn, mlp      visionNormAt
	norm           bool
	qkv, o, up, dn visionMat
	global         bool
}

type layaHeadAt struct {
	norm1, norm2   visionNormAt
	qkv, o, up, dn visionMat
}

// A LayaPipeline is the model resident on a device.
type LayaPipeline struct {
	d *Device
	s LayaShape

	par     []float32
	zeros   uint32 // Dim zeros in par, the bias of a norm that has none
	global  uint32 // where the rotation tables start in par
	local   uint32
	embed   visionMat // the token embedding, a row a token
	embNorm visionNormAt
	scoreN  visionNormAt // the scorer's norm and hidden layer
	scoreH  visionMat
	final   visionNormAt
	kinds   uint32 // the three rows of the type embedding
	enc     []layaEncoderAt
	head    []layaHeadAt
	mats    []visionMat
	total   int
	built   bool
	weights *Buffer
	parBuf  *Buffer

	normPipe, splitNormPipe, bigPipe, smallPipe, reducePipe, rowsPipe, attnPipe, actPipe *Pipeline
	queryTile                                                                            int
	// scalar is a card without the matrix cores: both tiles are
	// shaders/laya_mm_scalar.comp's, sixty-four square.
	scalar bool

	sc    *layaScratch
	trace map[string][]float32
	tl    *Timeline // set by Profile
}

// Profile makes every Encode after it stamp the card's clock between its
// stages, which Report then reads.
func (p *LayaPipeline) Profile() error {
	if p.tl != nil {
		return nil
	}
	tl, err := p.d.NewTimeline(4096)
	if err != nil {
		return err
	}
	p.tl = tl
	return nil
}

// Report is where the last profiled Encode spent the card's time, scaled to
// the wall clock the caller measured around it.
func (p *LayaPipeline) Report(total time.Duration) (string, error) {
	if p.tl == nil {
		return "", fmt.Errorf("vk: laya was not profiled")
	}
	return p.tl.Report(total)
}

func (p *LayaPipeline) stamp(r *Recorder, label string) {
	if p.tl != nil {
		p.tl.Stamp(r, label)
	}
}

// layaProduct is one product's sets, from one operand to one destination:
// the large tile, the small one, the small one split along the shared
// dimension into the partial sums, and the reduce that finishes those.
type layaProduct struct{ big, small, split, reduce *Set }

type layaScratch struct {
	n, cols, units int

	x, ops, gs, qkv, pos, kind, unit, idx, sel, partial *Buffer
	posin, kindin, unitin, idxin, back, tap             *Buffer

	norm, normX, normSel, split, kindAdd, attn, embed, gather *Set
	opsQKV, opsX, opsGS, gsX                                  layaProduct
}

type layaMMPush struct {
	outputs, inputs, cols, base, bias, mode, lo, slices, kper, stride, olo uint32
}

type layaNormPush struct {
	dim, gain, bias uint32
	eps             float32
	lo              uint32
}

type layaActPush struct{ count, op, width, at uint32 }

type layaRowsPush struct{ count, op, dim, base uint32 }

type layaAttnPush struct {
	dim, head uint32
	scale     float32
	window    uint32
	lo        uint32
	table     uint32 // in vec4, or ~0 for no rotation
}

// NewLayaPipeline creates the kernels. The weights arrive afterwards and
// nothing reaches the card until Prepare.
func NewLayaPipeline(d *Device, s LayaShape) (*LayaPipeline, error) {
	if s.HeadDim != nomicFlashHead {
		return nil, fmt.Errorf("vk: laya's attention is written for heads of %d, this one is %d", nomicFlashHead, s.HeadDim)
	}
	// Every matrix is a whole number of large tiles tall and every shared
	// dimension a whole number of the kernel's steps: shaders/laya_mm.comp
	// checks neither.
	for _, w := range []int{s.Dim, 3 * s.Dim, 2 * s.FF, s.HeadFF} {
		if w%layaBig != 0 {
			return nil, fmt.Errorf("vk: laya's products want every matrix a multiple of %d rows, and one is %d", layaBig, w)
		}
	}
	for _, w := range []int{s.Dim, s.FF, s.HeadFF} {
		if w%32 != 0 {
			return nil, fmt.Errorf("vk: laya's products read the shared dimension thirty-two at a time, and one is %d", w)
		}
	}
	p := &LayaPipeline{d: d, s: s, queryTile: layaQueryTile, scalar: !d.Coopmat()}
	big, small, wave := layaMMSPIRV, layaMMSmallSPIRV, uint32(layaWave)
	if p.scalar {
		big, small, wave = layaMMScalarSPIRV, layaMMScalarSPIRV, 0
	}
	for _, c := range []struct {
		dst      **Pipeline
		spirv    []byte
		bindings int
		push     uintptr
		wave     uint32
	}{
		{&p.normPipe, visionNormSPIRV, 3, unsafe.Sizeof(visionNormPush{}), 0},
		{&p.splitNormPipe, layaNormSPIRV, 3, unsafe.Sizeof(layaNormPush{}), 0},
		{&p.bigPipe, big, 5, unsafe.Sizeof(layaMMPush{}), wave},
		{&p.smallPipe, small, 5, unsafe.Sizeof(layaMMPush{}), wave},
		{&p.reducePipe, layaReduceSPIRV, 4, unsafe.Sizeof(layaMMPush{}), 0},
		{&p.rowsPipe, layaRowsSPIRV, 4, unsafe.Sizeof(layaRowsPush{}), 0},
		{&p.attnPipe, layaAttnSPIRV, 5, unsafe.Sizeof(layaAttnPush{}), 0},
		{&p.actPipe, layaActSPIRV, 4, unsafe.Sizeof(layaActPush{}), 0},
	} {
		pipe, err := d.newPipeline(c.spirv, c.bindings, uint32(c.push), c.wave, nil)
		if err != nil {
			p.Close()
			return nil, err
		}
		*c.dst = pipe
	}
	p.zeros = uint32(len(p.par))
	p.par = append(p.par, make([]float32, s.Dim)...)
	return p, nil
}

// SetRoPE takes the rotation tables of the global and the local blocks:
// MaxLen positions of HeadDim floats each, the cosines of half a head then
// its sines.
func (p *LayaPipeline) SetRoPE(global, local []float32) error {
	want := p.s.MaxLen * p.s.HeadDim
	if len(global) != want || len(local) != want {
		return fmt.Errorf("vk: a rotation table of %d positions is %d floats, given %d and %d",
			p.s.MaxLen, want, len(global), len(local))
	}
	// The attention reads the tables four floats at a time.
	p.par = append(p.par, make([]float32, (4-len(p.par)%4)%4)...)
	p.global = uint32(len(p.par))
	p.par = append(p.par, global...)
	p.local = uint32(len(p.par))
	p.par = append(p.par, local...)
	return nil
}

func (p *LayaPipeline) keepNorm(gain, bias []float32) (visionNormAt, error) {
	if len(gain) != p.s.Dim || (bias != nil && len(bias) != p.s.Dim) {
		return visionNormAt{}, fmt.Errorf("vk: a norm of %d was given a gain of %d and a bias of %d",
			p.s.Dim, len(gain), len(bias))
	}
	at := visionNormAt{gain: uint32(len(p.par)), bias: p.zeros}
	p.par = append(p.par, gain...)
	if bias != nil {
		at.bias = uint32(len(p.par))
		p.par = append(p.par, bias...)
	}
	return at, nil
}

func (p *LayaPipeline) keepLinear(l NomicLinear, outputs, inputs int) (visionMat, error) {
	if want := outputs * inputs * 2; len(l.W) != want {
		return visionMat{}, fmt.Errorf("vk: an fp16 matrix of %d by %d is %d bytes, given %d",
			outputs, inputs, want, len(l.W))
	}
	m := visionMat{src: l.W, outputs: outputs, inputs: inputs, bias: ^uint32(0), base: uint32(p.total / 2)}
	if l.Bias != nil {
		if len(l.Bias) != outputs {
			return visionMat{}, fmt.Errorf("vk: a projection onto %d wants that many biases, given %d", outputs, len(l.Bias))
		}
		m.bias = uint32(len(p.par))
		p.par = append(p.par, l.Bias...)
	}
	p.total += len(l.W)
	p.mats = append(p.mats, m)
	return m, nil
}

// SetEmbedding takes the token embedding, fp16, a row of Dim a token, and the
// norm the encoder puts over it, which has no bias.
func (p *LayaPipeline) SetEmbedding(table []byte, gain []float32) error {
	vocab := len(table) / (2 * p.s.Dim)
	if vocab == 0 || vocab*2*p.s.Dim != len(table) {
		return fmt.Errorf("vk: an embedding of rows of %d is a whole number of them, given %d bytes", p.s.Dim, len(table))
	}
	var err error
	if p.embed, err = p.keepLinear(NomicLinear{W: table}, vocab, p.s.Dim); err != nil {
		return err
	}
	p.embNorm, err = p.keepNorm(gain, nil)
	return err
}

// SetScorer takes the first half of the scorer: its norm, with a bias, and
// its hidden layer, whose GELU is what Encode hands back for the marked
// positions. The last layer is one row, which the host takes.
func (p *LayaPipeline) SetScorer(gain, bias []float32, hid NomicLinear) error {
	var err error
	if p.scoreN, err = p.keepNorm(gain, bias); err != nil {
		return err
	}
	p.scoreH, err = p.keepLinear(hid, p.s.Dim, p.s.Dim)
	return err
}

// interleave puts the rows of ModernBERT's up projection in the order the
// gated epilogue reads them: input row j, then gate row j, which is row
// FF+j of the matrix as it is stored.
func interleave(w []byte, ff, inputs int) []byte {
	out := make([]byte, len(w))
	row := inputs * 2
	for j := 0; j < ff; j++ {
		copy(out[(2*j)*row:(2*j+1)*row], w[j*row:(j+1)*row])
		copy(out[(2*j+1)*row:(2*j+2)*row], w[(ff+j)*row:(ff+j+1)*row])
	}
	return out
}

// AddEncoderBlock takes one block of the encoder, in order.
func (p *LayaPipeline) AddEncoderBlock(b LayaEncoderBlock) error {
	s := p.s
	at := layaEncoderAt{global: b.Global, norm: b.AttnGain != nil}
	var err error
	if at.norm {
		if at.attn, err = p.keepNorm(b.AttnGain, nil); err != nil {
			return err
		}
	}
	if at.mlp, err = p.keepNorm(b.MLPGain, nil); err != nil {
		return err
	}
	if at.qkv, err = p.keepLinear(b.QKV, 3*s.Dim, s.Dim); err != nil {
		return err
	}
	if at.o, err = p.keepLinear(b.O, s.Dim, s.Dim); err != nil {
		return err
	}
	if len(b.Up.W) != 2*s.FF*s.Dim*2 {
		return fmt.Errorf("vk: the up projection is %d bytes, want %d", len(b.Up.W), 2*s.FF*s.Dim*2)
	}
	up := NomicLinear{W: interleave(b.Up.W, s.FF, s.Dim)}
	if at.up, err = p.keepLinear(up, 2*s.FF, s.Dim); err != nil {
		return err
	}
	if at.dn, err = p.keepLinear(b.Down, s.Dim, s.FF); err != nil {
		return err
	}
	p.enc = append(p.enc, at)
	return nil
}

// SetFinalNorm takes the encoder's last norm, which has no bias.
func (p *LayaPipeline) SetFinalNorm(gain []float32) error {
	var err error
	p.final, err = p.keepNorm(gain, nil)
	return err
}

// SetKinds takes the type embedding: one row of Dim for each kind of
// question, choice, score and noul, back to back.
func (p *LayaPipeline) SetKinds(rows []float32) error {
	if len(rows) != 3*p.s.Dim {
		return fmt.Errorf("vk: the type embedding is three rows of %d, given %d floats", p.s.Dim, len(rows))
	}
	p.kinds = uint32(len(p.par))
	p.par = append(p.par, rows...)
	return nil
}

// AddHeadBlock takes one layer of the head, in order.
func (p *LayaPipeline) AddHeadBlock(b LayaHeadBlock) error {
	s := p.s
	var at layaHeadAt
	var err error
	if at.norm1, err = p.keepNorm(b.Norm1Gain, b.Norm1Bias); err != nil {
		return err
	}
	if at.norm2, err = p.keepNorm(b.Norm2Gain, b.Norm2Bias); err != nil {
		return err
	}
	if at.qkv, err = p.keepLinear(b.QKV, 3*s.Dim, s.Dim); err != nil {
		return err
	}
	if at.o, err = p.keepLinear(b.O, s.Dim, s.Dim); err != nil {
		return err
	}
	if at.up, err = p.keepLinear(b.Up, s.HeadFF, s.Dim); err != nil {
		return err
	}
	if at.dn, err = p.keepLinear(b.Down, s.Dim, s.HeadFF); err != nil {
		return err
	}
	p.head = append(p.head, at)
	return nil
}

// Prepare uploads everything. The model is under a gigabyte in fp16, so it
// is resident or it is nothing.
func (p *LayaPipeline) Prepare() error {
	if p.built {
		return nil
	}
	var err error
	if p.parBuf, err = p.d.Upload(asBytes(p.par)); err != nil {
		return err
	}
	if p.weights, err = p.d.Local(p.total, bufferUsageStorage|bufferUsageTransferDst); err != nil {
		return fmt.Errorf("vk: the card has no room for laya's %d MiB: %w", p.total>>20, err)
	}
	const chunk = 64 << 20
	stage, err := p.d.Host(chunk, bufferUsageTransferSrc)
	if err != nil {
		return err
	}
	defer stage.Close()
	for i := range p.mats {
		m := &p.mats[i]
		at := int(m.base) * 2
		for off := 0; off < len(m.src); off += chunk {
			n := min(chunk, len(m.src)-off)
			copy(stage.Bytes(), m.src[off:off+n])
			if err := p.d.Submit(func(r *Recorder) { r.Copy(p.weights, at+off, stage, n) }); err != nil {
				return err
			}
		}
		m.src = nil
	}
	p.built = true
	return nil
}

// Trace makes the next Encode keep x after every block and after the
// encoder's last norm, under ref/laya/dump.py's names, which Waypoint hands
// back. It submits a recording a block.
func (p *LayaPipeline) Trace() { p.trace = map[string][]float32{} }

// Waypoint is what the last traced Encode left under that name, or nil.
func (p *LayaPipeline) Waypoint(name string) []float32 { return p.trace[name] }

// Encode runs the embedding, the encoder and the head over one pass.
//
// ids is each position's token; pos is each position's place in its
// sequence, seg its sequence's start and length, two a position, and kind its
// question's kind. rows are the positions whose final state the caller wants,
// and out receives them, Dim floats a row, in that order; marks are the
// positions the scorer reads, and hid receives its hidden layer for each of
// them, Dim floats a mark. The host scores a few positions of a pass, and the
// rest need not cross back.
func (p *LayaPipeline) Encode(ids, pos, seg, kind, rows, marks []uint32, out, hid []float32) error {
	if !p.built {
		return fmt.Errorf("vk: laya has not been prepared")
	}
	if p.embed.outputs == 0 {
		return fmt.Errorf("vk: laya has no embedding")
	}
	s := p.s
	n := len(ids)
	if n == 0 || len(pos) != n || len(seg) != 2*n || len(kind) != n || len(rows)+len(marks) > n ||
		len(out) != len(rows)*s.Dim || len(hid) != len(marks)*s.Dim {
		return fmt.Errorf("vk: a pass of %d positions wants a place, two bounds and a kind each, and a row of %d for each of %d rows and %d marks asked",
			n, s.Dim, len(rows), len(marks))
	}
	if len(marks) > 0 && p.scoreH.outputs == 0 {
		return fmt.Errorf("vk: laya has no scorer")
	}
	for t, at := range pos {
		if int(at) >= s.MaxLen {
			return fmt.Errorf("vk: position %d is past the rotation tables' %d", at, s.MaxLen)
		}
		if int(ids[t]) >= p.embed.outputs {
			return fmt.Errorf("vk: token %d is outside a vocabulary of %d", ids[t], p.embed.outputs)
		}
	}
	for _, r := range append(append([]uint32(nil), rows...), marks...) {
		if int(r) >= n {
			return fmt.Errorf("vk: row %d asked of a pass of %d", r, n)
		}
	}
	if err := p.scratchFor(n); err != nil {
		return err
	}
	sc := p.sc
	copy(sc.posin.Uints(), pos)
	copy(sc.kindin.Uints(), kind)
	copy(sc.idxin.Uints(), ids)
	// The gathered rows are the marks, then the rows: the scorer's norm
	// reads the first of them.
	copy(sc.idxin.Uints()[n:], marks)
	copy(sc.idxin.Uints()[n+len(marks):], rows)
	units := sc.unitin.Uints()[:0]
	for t := 0; t < n; {
		start, length := seg[2*t], seg[2*t+1]
		if int(start) != t || length == 0 || t+int(length) > n {
			return fmt.Errorf("vk: position %d says its sequence starts at %d and is %d long", t, start, length)
		}
		for q0 := uint32(0); q0 < length; q0 += uint32(p.queryTile) {
			units = append(units, start, length, q0, 0)
		}
		t += int(length)
	}
	sc.units = len(units) / 4

	var pending []func(*Recorder)
	flush := func() error {
		if len(pending) == 0 {
			return nil
		}
		steps := pending
		pending = nil
		return p.d.Submit(func(r *Recorder) {
			for _, step := range steps {
				step(r)
			}
		})
	}
	keep := func(name string) error {
		if p.trace == nil {
			return nil
		}
		pending = append(pending, func(r *Recorder) { r.Copy(sc.tap, 0, sc.x, n*s.Dim*4) })
		if err := flush(); err != nil {
			return err
		}
		p.trace[name] = append([]float32(nil), sc.tap.Floats()[:n*s.Dim]...)
		return nil
	}

	pending = append(pending, func(r *Recorder) {
		if p.tl != nil {
			p.tl.Reset(r)
			p.stamp(r, "start")
		}
		r.Copy(sc.pos, 0, sc.posin, n*4)
		r.Copy(sc.kind, 0, sc.kindin, n*4)
		r.Copy(sc.idx, 0, sc.idxin, (n+len(marks)+len(rows))*4)
		r.Copy(sc.unit, 0, sc.unitin, sc.units*16)
		r.Barrier()
		p.stamp(r, "upload")
		emb := layaRowsPush{count: uint32(n * s.Dim), op: layaEmbed, dim: uint32(s.Dim), base: p.embed.base}
		r.Dispatch(sc.embed, uint32((n*s.Dim+255)/256), unsafe.Pointer(&emb))
		r.Barrier()
		norm := visionNormPush{dim: uint32(s.Dim), gain: p.embNorm.gain, bias: p.embNorm.bias, eps: s.Eps}
		r.Dispatch(sc.normX, uint32(n), unsafe.Pointer(&norm))
		r.Barrier()
		p.stamp(r, "embed")
	})

	for i := range p.enc {
		b := &p.enc[i]
		pending = append(pending, func(r *Recorder) {
			if b.norm {
				p.norm(r, b.attn, n)
			} else {
				p.act(r, sc.split, n*s.Dim, layaSplit, 0, uint32(sc.cols*s.Dim))
			}
			r.Barrier()
			p.stamp(r, "norm")
			table, window := p.local, s.Window
			if b.global {
				table, window = p.global, 0
			}
			p.attention(r, &b.qkv, &b.o, table, window, n)

			p.norm(r, b.mlp, n)
			r.Barrier()
			p.stamp(r, "norm")
			p.product(r, &sc.opsGS, &b.up, n, layaGeGLU)
			r.Barrier()
			p.stamp(r, "up")
			p.product(r, &sc.gsX, &b.dn, n, layaAdd)
			r.Barrier()
			p.stamp(r, "down")
		})
		if err := keep(fmt.Sprintf("layer-%d", i)); err != nil {
			return fmt.Errorf("vk: laya failed at block %d: %w", i, err)
		}
	}
	pending = append(pending, func(r *Recorder) {
		push := visionNormPush{dim: uint32(s.Dim), gain: p.final.gain, bias: p.final.bias, eps: s.Eps}
		r.Dispatch(sc.normX, uint32(n), unsafe.Pointer(&push))
		r.Barrier()
	})
	if err := keep("encoded"); err != nil {
		return err
	}

	pending = append(pending, func(r *Recorder) {
		p.act(r, sc.kindAdd, n*s.Dim, layaKind, s.Dim, p.kinds)
		r.Barrier()
	})
	for i := range p.head {
		b := &p.head[i]
		pending = append(pending, func(r *Recorder) {
			p.norm(r, b.norm1, n)
			r.Barrier()
			p.attention(r, &b.qkv, &b.o, ^uint32(0), 0, n)
			p.norm(r, b.norm2, n)
			r.Barrier()
			p.product(r, &sc.opsGS, &b.up, n, layaReLU)
			r.Barrier()
			p.product(r, &sc.gsX, &b.dn, n, layaAdd)
			r.Barrier()
			p.stamp(r, "head")
		})
		if err := keep(fmt.Sprintf("head-%d", i)); err != nil {
			return fmt.Errorf("vk: laya failed at head layer %d: %w", i, err)
		}
	}
	pending = append(pending, func(r *Recorder) {
		m := len(marks)
		if total := m + len(rows); total > 0 {
			g := layaRowsPush{count: uint32(total * s.Dim), op: layaGather, dim: uint32(s.Dim), base: uint32(n)}
			r.Dispatch(sc.gather, uint32((total*s.Dim+255)/256), unsafe.Pointer(&g))
			r.Barrier()
		}
		if m > 0 {
			push := layaNormPush{dim: uint32(s.Dim), gain: p.scoreN.gain, bias: p.scoreN.bias, eps: s.Eps,
				lo: uint32(sc.cols * s.Dim)}
			r.Dispatch(sc.normSel, uint32(m), unsafe.Pointer(&push))
			r.Barrier()
			p.product(r, &sc.opsQKV, &p.scoreH, m, layaGELU)
			r.Barrier()
			r.Copy(sc.back, 0, sc.qkv, m*s.Dim*4)
		}
		if len(rows) > 0 {
			r.CopyFrom(sc.back, m*s.Dim*4, sc.sel, m*s.Dim*4, len(rows)*s.Dim*4)
		}
		p.stamp(r, "readback")
	})
	if err := flush(); err != nil {
		return fmt.Errorf("vk: laya failed reading back: %w", err)
	}
	back := sc.back.Floats()
	copy(hid, back[:len(marks)*s.Dim])
	copy(out, back[len(marks)*s.Dim:(len(marks)+len(rows))*s.Dim])
	return nil
}

// attention is the first half of a block, from its split operand to x: the
// fused projection, the attention, which turns the queries and keys by table
// (or not, for ~0) as it reads them, and the output projection added into x.
func (p *LayaPipeline) attention(r *Recorder, wqkv, wo *visionMat, table uint32, window, n int) {
	s, sc := p.s, p.sc
	p.product(r, &sc.opsQKV, wqkv, n, layaStore)
	r.Barrier()
	p.stamp(r, "qkv")
	attn := layaAttnPush{dim: uint32(s.Dim), head: uint32(s.HeadDim),
		scale: float32(1 / math.Sqrt(float64(s.HeadDim))), window: uint32(window), lo: uint32(sc.cols * s.Dim / 4),
		table: ^uint32(0)}
	if table != ^uint32(0) {
		attn.table = table / 4
	}
	r.DispatchColumns(sc.attn, uint32(sc.units), uint32(s.Heads), unsafe.Pointer(&attn))
	r.Barrier()
	p.stamp(r, "attention")
	p.product(r, &sc.opsX, wo, n, layaAdd)
	r.Barrier()
	p.stamp(r, "o")
}

// norm writes norm(x) into the operand planes.
func (p *LayaPipeline) norm(r *Recorder, at visionNormAt, n int) {
	push := layaNormPush{dim: uint32(p.s.Dim), gain: at.gain, bias: at.bias, eps: p.s.Eps, lo: uint32(p.sc.cols * p.s.Dim)}
	r.Dispatch(p.sc.norm, uint32(n), unsafe.Pointer(&push))
}

func (p *LayaPipeline) act(r *Recorder, set *Set, count int, op uint32, width int, at uint32) {
	push := layaActPush{count: uint32(count), op: op, width: uint32(width), at: at}
	r.Dispatch(set, uint32((count+255)/256), unsafe.Pointer(&push))
}

// Below this many workgroups a product leaves the card idle: sixty-four
// compute units, each of which holds two of the large tiles at once.
const (
	layaBigWorkgroups   = 96
	layaSmallWorkgroups = 128
)

// product is m times the operand, finished by the epilogue mode: the large
// tile when the pass makes enough of them, the small one otherwise, and the
// small one split along the shared dimension when even that leaves the card
// idle.
func (p *LayaPipeline) product(r *Recorder, sets *layaProduct, m *visionMat, cols int, mode uint32) {
	sc := p.sc
	// The operand's low plane: the operand planes are sc.cols columns of
	// their own width, and the gated ones as wide as the widest of the two
	// feed forwards.
	gsLo := uint32(sc.cols * max(p.s.FF, p.s.HeadFF))
	push := layaMMPush{outputs: uint32(m.outputs), inputs: uint32(m.inputs), cols: uint32(cols),
		base: m.base, bias: m.bias, mode: mode, lo: uint32(sc.cols * m.inputs), olo: gsLo}
	if sets == &sc.gsX {
		push.lo = gsLo
	}
	if tiles := (m.outputs / layaBig) * ((cols + layaBig - 1) / layaBig); !p.scalar && tiles >= layaBigWorkgroups {
		r.Dispatch(sets.big, uint32(tiles), unsafe.Pointer(&push))
		return
	}
	groups := (m.outputs / laySmall) * ((cols + laySmall - 1) / laySmall)
	if groups < layaSmallWorkgroups && m.inputs >= 2*layaSlice {
		slices := min((layaSmallWorkgroups+groups-1)/groups, m.inputs/layaSlice)
		kper := (m.inputs + slices - 1) / slices
		kper = (kper + 31) / 32 * 32
		slices = (m.inputs + kper - 1) / kper
		if slices >= 2 && slices*cols*m.outputs <= nomicPartialFloats {
			push.slices, push.kper, push.stride = uint32(slices), uint32(kper), uint32(cols*m.outputs)
			r.Dispatch(sets.split, uint32(groups*slices), unsafe.Pointer(&push))
			r.Barrier()
			count := cols * m.outputs
			if mode == layaGeGLU {
				count /= 2
			}
			r.Dispatch(sets.reduce, uint32((count+255)/256), unsafe.Pointer(&push))
			return
		}
	}
	r.Dispatch(sets.small, uint32(groups), unsafe.Pointer(&push))
}

// scratchFor makes the buffers a pass of n positions needs, or keeps the ones
// a pass at least as wide left behind.
func (p *LayaPipeline) scratchFor(n int) error {
	if p.sc != nil && p.sc.n >= n && (p.trace == nil || p.sc.tap != nil) {
		return nil
	}
	if p.sc != nil {
		p.sc.close()
		p.sc = nil
	}
	s := p.s
	size := max(n, 64)
	// The operand planes are read a whole large tile of columns at a time.
	cols := (size + layaBig - 1) / layaBig * layaBig
	ff := max(s.FF, s.HeadFF)
	sc := &layaScratch{n: size, cols: cols}
	fail := func(err error) error {
		sc.close()
		return err
	}
	for _, c := range []struct {
		b     **Buffer
		bytes int
	}{
		{&sc.x, size * s.Dim * 4},
		{&sc.ops, 2 * cols * s.Dim * 2},
		{&sc.gs, 2 * cols * ff * 2},
		{&sc.qkv, size * 3 * s.Dim * 4},
		{&sc.sel, size * s.Dim * 4},
		{&sc.idx, 2 * size * 4},
		{&sc.pos, size * 4},
		{&sc.kind, size * 4},
		{&sc.unit, 4 * size * 4},
		{&sc.partial, nomicPartialFloats * 4},
	} {
		v, err := p.d.Local(c.bytes, bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst)
		if err != nil {
			return fail(fmt.Errorf("vk: the card has no room for a pass of %d positions: %w", size, err))
		}
		*c.b = v
	}
	// The planes' columns past a pass are read and never written: zeros, so
	// that nothing a product meets there is other than a number.
	if err := p.d.Submit(func(r *Recorder) {
		r.Fill(sc.ops, 0)
		r.Fill(sc.gs, 0)
	}); err != nil {
		return fail(err)
	}
	var err error
	for _, c := range []struct {
		b     **Buffer
		bytes int
		back  bool
	}{
		{&sc.idxin, 2 * size * 4, false},
		{&sc.posin, size * 4, false},
		{&sc.kindin, size * 4, false},
		{&sc.unitin, 4 * size * 4, false},
		{&sc.back, size * s.Dim * 4, true},
	} {
		if c.back {
			*c.b, err = p.d.Readback(c.bytes, bufferUsageTransferDst)
		} else {
			*c.b, err = p.d.Host(c.bytes, bufferUsageTransferSrc)
		}
		if err != nil {
			return fail(err)
		}
	}
	if p.trace != nil {
		if sc.tap, err = p.d.Readback(size*s.Dim*4, bufferUsageTransferDst); err != nil {
			return fail(err)
		}
	}
	for _, c := range []struct {
		set     **Set
		pipe    *Pipeline
		buffers []*Buffer
	}{
		{&sc.norm, p.splitNormPipe, []*Buffer{sc.x, p.parBuf, sc.ops}},
		{&sc.normX, p.normPipe, []*Buffer{sc.x, p.parBuf, sc.x}},
		{&sc.normSel, p.splitNormPipe, []*Buffer{sc.sel, p.parBuf, sc.ops}},
		{&sc.split, p.actPipe, []*Buffer{sc.x, sc.ops, p.parBuf, sc.kind}},
		{&sc.kindAdd, p.actPipe, []*Buffer{sc.x, sc.ops, p.parBuf, sc.kind}},
		{&sc.attn, p.attnPipe, []*Buffer{sc.qkv, sc.pos, p.parBuf, sc.ops, sc.unit}},
		{&sc.embed, p.rowsPipe, []*Buffer{p.weights, sc.x, sc.idx, sc.sel}},
		{&sc.gather, p.rowsPipe, []*Buffer{p.weights, sc.x, sc.idx, sc.sel}},
	} {
		set, err := c.pipe.NewSet(c.buffers)
		if err != nil {
			return fail(err)
		}
		*c.set = set
	}
	for _, c := range []struct {
		dst         *layaProduct
		in, out, hl *Buffer
	}{
		{&sc.opsQKV, sc.ops, sc.qkv, sc.gs},
		{&sc.opsX, sc.ops, sc.x, sc.gs},
		{&sc.opsGS, sc.ops, sc.x, sc.gs},
		{&sc.gsX, sc.gs, sc.x, sc.ops},
	} {
		if c.dst.big, err = p.bigPipe.NewSet([]*Buffer{p.weights, c.in, p.parBuf, c.out, c.hl}); err != nil {
			return fail(err)
		}
		if c.dst.small, err = p.smallPipe.NewSet([]*Buffer{p.weights, c.in, p.parBuf, c.out, c.hl}); err != nil {
			return fail(err)
		}
		if c.dst.split, err = p.smallPipe.NewSet([]*Buffer{p.weights, c.in, p.parBuf, sc.partial, c.hl}); err != nil {
			return fail(err)
		}
		if c.dst.reduce, err = p.reducePipe.NewSet([]*Buffer{sc.partial, p.parBuf, c.out, c.hl}); err != nil {
			return fail(err)
		}
	}
	p.sc = sc
	return nil
}

func (sc *layaScratch) close() {
	sets := []**Set{&sc.norm, &sc.normX, &sc.normSel, &sc.split, &sc.kindAdd, &sc.attn, &sc.embed, &sc.gather}
	for _, g := range []*layaProduct{&sc.opsQKV, &sc.opsX, &sc.opsGS, &sc.gsX} {
		sets = append(sets, &g.big, &g.small, &g.split, &g.reduce)
	}
	for _, s := range sets {
		if *s != nil {
			(*s).Close()
			*s = nil
		}
	}
	for _, b := range []**Buffer{&sc.x, &sc.ops, &sc.gs, &sc.qkv, &sc.sel, &sc.idx, &sc.pos, &sc.kind, &sc.unit,
		&sc.partial, &sc.idxin, &sc.posin, &sc.kindin, &sc.unitin, &sc.back, &sc.tap} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}

// Close releases everything laya holds on the device.
func (p *LayaPipeline) Close() {
	if p.sc != nil {
		p.sc.close()
		p.sc = nil
	}
	if p.tl != nil {
		p.tl.Close()
		p.tl = nil
	}
	for _, pipe := range []**Pipeline{&p.normPipe, &p.splitNormPipe, &p.bigPipe, &p.smallPipe, &p.reducePipe,
		&p.rowsPipe, &p.attnPipe, &p.actPipe} {
		if *pipe != nil {
			(*pipe).Close()
			*pipe = nil
		}
	}
	for _, b := range []**Buffer{&p.parBuf, &p.weights} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	p.built = false
}
