package vk

// nomic-embed-text-v2-moe on the card.
//
// An encoder is the vision tower's kind of work — every position at once, no
// cache, fp16 weights, norms that subtract a mean — and it borrows the tower's
// norm and its GELU-and-residual kernel. The rest is its own. A pass carries
// several texts, so the attention is bounded by each position's text and the
// rotation reads each position's place in it. The product runs on the matrix
// cores when the device took them, and the attention takes a tile of queries
// at a time. The blocks are post-norm, which is the same kernels in another
// order. And every odd block is a mixture: a router, then per expert a product
// that reads only the positions that chose it and one that adds its answer
// back to them — all eight experts in one dispatch each way, off a table
// the host writes, and a last kernel that sums each position's slots.
//
// The router's choice is made on the card, by shaders/nomic_route.comp, which
// also writes the table the grouped products walk, so a pass is one
// submission. A traced pass, and one too wide for that kernel, route on the
// host instead: the logits come back, the host picks as the processor's path
// does, and the table goes up — one submission per mixture block.

import (
	_ "embed"
	"fmt"
	"math"
	"time"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_gemm.comp -o shaders/nomic_gemm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DGROUPED shaders/nomic_gemm.comp -o shaders/nomic_gemm_grouped.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_mm.comp -o shaders/nomic_mm.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DGROUPED shaders/nomic_mm.comp -o shaders/nomic_mm_grouped.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_combine.comp -o shaders/nomic_combine.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_mm_big.comp -o shaders/nomic_mm_big.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_route.comp -o shaders/nomic_route.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_flash_coop.comp -o shaders/nomic_flash_coop.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_reduce.comp -o shaders/nomic_reduce.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute -DGROUPED shaders/nomic_mm_big.comp -o shaders/nomic_mm_big_grouped.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_rope.comp -o shaders/nomic_rope.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_attn.comp -o shaders/nomic_attn.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_flash.comp -o shaders/nomic_flash.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/nomic_router.comp -o shaders/nomic_router.spv

//go:embed shaders/nomic_gemm.spv
var nomicGemmSPIRV []byte

//go:embed shaders/nomic_gemm_grouped.spv
var nomicGroupedSPIRV []byte

//go:embed shaders/nomic_mm.spv
var nomicMMSPIRV []byte

//go:embed shaders/nomic_mm_grouped.spv
var nomicMMGroupedSPIRV []byte

//go:embed shaders/nomic_combine.spv
var nomicCombineSPIRV []byte

//go:embed shaders/nomic_mm_big.spv
var nomicMMBigSPIRV []byte

//go:embed shaders/nomic_route.spv
var nomicRouteSPIRV []byte

//go:embed shaders/nomic_flash_coop.spv
var nomicFlashCoopSPIRV []byte

//go:embed shaders/nomic_reduce.spv
var nomicReduceSPIRV []byte

// A product that makes fewer than nomicSplitWorkgroups tiles is split along
// its shared dimension until it makes that many, each slice at least 256 of
// it. The partial sums of every slice have to fit nomicPartialFloats, which
// only a narrow pass — the only kind that is split — ever asks of it.
const (
	nomicSplitWorkgroups = 128
	nomicPartialFloats   = 16 << 20
)

// nomicRouteSlots is the most choices shaders/nomic_route.comp holds in its
// shared memory, positions times experts used. A wider pass routes on the
// host, as a traced one does.
const nomicRouteSlots = 8192

// nomicRouteExperts is the most experts it counts.
const nomicRouteExperts = 16

//go:embed shaders/nomic_mm_big_grouped.spv
var nomicMMBigGroupedSPIRV []byte

// nomicBigWorkgroups is how many of the large tiles a product has to make
// before it takes them. Fewer leaves the card's sixty-four compute units
// waiting on a handful of workgroups, and the small tile, four times as
// many of them, is then the faster of the two.
const nomicBigWorkgroups = 16

//go:embed shaders/nomic_rope.spv
var nomicRopeSPIRV []byte

//go:embed shaders/nomic_attn.spv
var nomicAttnSPIRV []byte

//go:embed shaders/nomic_flash.spv
var nomicFlashSPIRV []byte

//go:embed shaders/nomic_router.spv
var nomicRouterSPIRV []byte

// nomicFlashHead is the head width shaders/nomic_flash.comp is written for.
// Another width takes shaders/nomic_attn.comp, a query at a time.
const nomicFlashHead = 64

// How many queries of one head a unit of the tiled attention takes:
// shaders/nomic_flash.comp's BR, and shaders/nomic_flash_coop.comp's.
const (
	nomicQueryTile     = 32
	nomicCoopQueryTile = 64
)

// NomicShape is the encoder's geometry.
type NomicShape struct {
	Dim, Heads, HeadDim, FF int
	Experts, Used           int
	Eps, RoPEBase           float32
}

// NomicLinear is one projection: fp16 weights, one row per output, and an F32
// bias or nil.
type NomicLinear struct {
	W    []byte
	Bias []float32
}

// NomicBlockData is one block. A dense block fills Up and Down; a mixture
// fills Router — Experts rows of Dim floats — and one pair of matrices an
// expert.
type NomicBlockData struct {
	QKV, O             NomicLinear
	AttnGain, AttnBias []float32
	Up, Down           NomicLinear
	Router             []float32
	UpExps, DownExps   []NomicLinear
	OutGain, OutBias   []float32
}

// NomicRouter is the host's choice for one mixture block. Given the logits,
// Experts a position, it returns the choices grouped by expert — expert 0's
// first, in position order — as two lists side by side: idx, the position
// each choice belongs to, and dest, which of that position's Used slots it
// fills, as position*Used+slot. wt is the weight of every slot, in slot order,
// Used a position; counts is how many choices each expert got.
type NomicRouter func(block int, logits []float32) (idx, dest []uint32, wt []float32, counts []int)

type nomicBlockAt struct {
	qkv, o, up, dn visionMat
	attn, out      visionNormAt
	mixture        bool
	router         uint32
	bases          uint32 // where each expert's two matrix bases sit in par, as bits
	upE, dnE       []visionMat
}

// A NomicPipeline is the encoder resident on a device.
type NomicPipeline struct {
	d *Device
	s NomicShape

	par    []float32
	emb    visionNormAt
	blocks []nomicBlockAt
	mats   []visionMat // every matrix, for the upload
	total  int         // bytes of weights
	built  bool

	parBuf, geluBuf, weights *Buffer

	normPipe, gemmPipe, groupPipe, combinePipe *Pipeline
	ropePipe, attnPipe, flashPipe, actPipe     *Pipeline
	routerPipe                                 *Pipeline
	bigPipe, bigGroupPipe                      *Pipeline // the large tile, on matrix cores only
	routePipe                                  *Pipeline
	reducePipe                                 *Pipeline // the split products', on matrix cores only

	sc    *nomicScratch
	trace map[string][]float32
	tl    *Timeline // set by Profile

	queryTile int // the queries a unit of the tiled attention takes
}

type nomicScratch struct {
	n     int
	units int // how many units of the tiled attention the current pass has

	x, h, qkv, q, k, mixed, tmp, up, ans, logits, pos, seg, idx, dest, wt, unit, table *Buffer
	xin, posin, segin, idxin, destin, wtin, unitin, tablein, back, logitsBack, tap     *Buffer

	normE, normH, normX, gQKV, gO, gUp, gDn, upE, dnE, combine *Set
	rope, attn, router, addX, addH, gelu                       *Set
	bQKV, bO, bUp, bDn, bUpE, bDnE                             *Set // the large tile's, or nil
	route                                                      *Set
	partial                                                    *Buffer
	// The split products write their partial sums to partial, and a reduce
	// adds them into where the product would have written. Nil without the
	// matrix cores.
	sQKV, sO, sUp, sDn, sUpE, sDnE, rQKV, rTmp, rUp, rAns *Set
}

type nomicCombinePush struct{ dim, used, count uint32 }

// nomicTableEntry is how many words one entry of the grouped products' table
// takes: shaders/nomic_mm.comp reads eight.
const nomicTableEntry = 8

type nomicGemmPush struct {
	outputs, inputs, patches, base, bias, index uint32
	slices, stride, kper                        uint32 // a split product's; zero for one that is not
}

type nomicReducePush struct{ count, outputs, slices, bias uint32 }

type nomicRopePush struct {
	dim, heads, head uint32
	base             float32
}

type nomicAttnPush struct {
	dim, head uint32
	scale     float32
}

type nomicRouterPush struct{ dim, experts, at uint32 }

type nomicRoutePush struct {
	n, experts, used, bases        uint32
	widthUp, widthDn, maxUp, maxDn uint32
}

// nomicTaps are what a traced pass keeps a block, under llama.cpp's names.
var nomicTaps = []string{"kqv_out", "ffn_inp", "l_out"}

// NewNomicPipeline creates the encoder's kernels. The weights arrive block by
// block and nothing reaches the card until Prepare.
func NewNomicPipeline(d *Device, s NomicShape) (*NomicPipeline, error) {
	if s.HeadDim > visionMaxHead {
		return nil, fmt.Errorf("vk: the encoder's attention is written for heads of at most %d, this one is %d",
			visionMaxHead, s.HeadDim)
	}
	if s.Dim%2 != 0 || s.FF%2 != 0 {
		return nil, fmt.Errorf("vk: an fp16 row is read two halves a word, so every width must be even")
	}
	p := &NomicPipeline{d: d, s: s}

	// The matrix cores when the device took them, and the same interface in
	// scalar arithmetic when it did not.
	gemm, grouped, wave := nomicGemmSPIRV, nomicGroupedSPIRV, uint32(0)
	if d.Coopmat() {
		gemm, grouped, wave = nomicMMSPIRV, nomicMMGroupedSPIRV, coopmatWave
	}
	attn, attnInto, attnWave := nomicAttnSPIRV, &p.attnPipe, uint32(0)
	if s.HeadDim == nomicFlashHead {
		attn, attnInto, p.queryTile = nomicFlashSPIRV, &p.flashPipe, nomicQueryTile
		if d.Coopmat() {
			attn, attnWave, p.queryTile = nomicFlashCoopSPIRV, coopmatWave, nomicCoopQueryTile
		}
	}
	for _, c := range []struct {
		dst      **Pipeline
		spirv    []byte
		bindings int
		push     uintptr
		wave     uint32
	}{
		{&p.normPipe, visionNormSPIRV, 3, unsafe.Sizeof(visionNormPush{}), 0},
		{&p.gemmPipe, gemm, 6, unsafe.Sizeof(nomicGemmPush{}), wave},
		{&p.groupPipe, grouped, 6, unsafe.Sizeof(nomicGemmPush{}), wave},
		{&p.combinePipe, nomicCombineSPIRV, 3, unsafe.Sizeof(nomicCombinePush{}), 0},
		{&p.ropePipe, nomicRopeSPIRV, 4, unsafe.Sizeof(nomicRopePush{}), 0},
		{attnInto, attn, 5, unsafe.Sizeof(nomicAttnPush{}), attnWave},
		{&p.actPipe, visionActSPIRV, 3, unsafe.Sizeof(visionActPush{}), 0},
		{&p.routerPipe, nomicRouterSPIRV, 3, unsafe.Sizeof(nomicRouterPush{}), 0},
		{&p.routePipe, nomicRouteSPIRV, 6, unsafe.Sizeof(nomicRoutePush{}), 0},
	} {
		pipe, err := d.newPipeline(c.spirv, c.bindings, uint32(c.push), c.wave, nil)
		if err != nil {
			p.Close()
			return nil, err
		}
		*c.dst = pipe
	}
	if d.Coopmat() {
		for _, c := range []struct {
			dst   **Pipeline
			spirv []byte
		}{{&p.bigPipe, nomicMMBigSPIRV}, {&p.bigGroupPipe, nomicMMBigGroupedSPIRV}, {&p.reducePipe, nomicReduceSPIRV}} {
			bindings, push, wave := 6, uint32(unsafe.Sizeof(nomicGemmPush{})), uint32(coopmatWave)
			if c.dst == &p.reducePipe {
				bindings, push, wave = 3, uint32(unsafe.Sizeof(nomicReducePush{})), 0
			}
			pipe, err := d.newPipeline(c.spirv, bindings, push, wave, nil)
			if err != nil {
				p.Close()
				return nil, err
			}
			*c.dst = pipe
		}
	}
	return p, nil
}

// split is how many slices a product making this many small tiles is cut
// into along its shared dimension, and how much of it each slice walks; one
// slice when it is wide enough, when the machine has no matrix cores, or when
// the partial sums would not fit.
func (p *NomicPipeline) split(workgroups, inputs, rows, outputs int) (slices, kper int) {
	if p.reducePipe == nil || workgroups >= nomicSplitWorkgroups || inputs < 512 {
		return 1, inputs
	}
	// One chunk of 256 a slice, which is what the kernels sum the shared
	// dimension in: the reduce then adds the chunks in the order a kernel that
	// was not split adds them, and the floats are the same. A slice of two
	// chunks would add them in pairs first — shaders/nomic_mm.comp says why
	// that matters.
	kper = 256
	s := (inputs + kper - 1) / kper
	if s < 2 || s*rows*outputs > nomicPartialFloats {
		return 1, inputs
	}
	return s, kper
}

func (p *NomicPipeline) reduce(r *Recorder, set *Set, rows, outputs, slices int, bias uint32) {
	push := nomicReducePush{count: uint32(rows * outputs), outputs: uint32(outputs), slices: uint32(slices), bias: bias}
	r.Dispatch(set, uint32((rows*outputs+255)/256), unsafe.Pointer(&push))
}

// big says whether a product this shape, making this many large tiles, takes
// them: the large tile reads eight halves and four floats at a time, which
// wants the shared dimension a whole number of thirty-twos.
func (p *NomicPipeline) big(inputs, tiles int) bool {
	return p.bigPipe != nil && inputs%32 == 0 && tiles >= nomicBigWorkgroups
}

// Profile makes every Encode after it stamp the card's clock between its
// stages, which Report then reads.
func (p *NomicPipeline) Profile() error {
	if p.tl != nil {
		return nil
	}
	tl, err := p.d.NewTimeline(2048)
	if err != nil {
		return err
	}
	p.tl = tl
	return nil
}

// Report is where the last profiled Encode spent the card's time, scaled to
// the wall clock the caller measured around it.
func (p *NomicPipeline) Report(total time.Duration) (string, error) {
	if p.tl == nil {
		return "", fmt.Errorf("vk: the encoder was not profiled")
	}
	return p.tl.Report(total)
}

func (p *NomicPipeline) stamp(r *Recorder, label string) {
	if p.tl != nil {
		p.tl.Stamp(r, label)
	}
}

func (p *NomicPipeline) keepNorm(gain, bias []float32) (visionNormAt, error) {
	if len(gain) != p.s.Dim || len(bias) != p.s.Dim {
		return visionNormAt{}, fmt.Errorf("vk: a norm of %d wants a gain and a bias of that width, given %d and %d",
			p.s.Dim, len(gain), len(bias))
	}
	at := visionNormAt{gain: uint32(len(p.par))}
	p.par = append(p.par, gain...)
	at.bias = uint32(len(p.par))
	p.par = append(p.par, bias...)
	return at, nil
}

func (p *NomicPipeline) keepLinear(l NomicLinear, outputs, inputs int) (visionMat, error) {
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

// SetEmbedNorm takes the norm the embeddings go through before block 0.
func (p *NomicPipeline) SetEmbedNorm(gain, bias []float32) error {
	var err error
	p.emb, err = p.keepNorm(gain, bias)
	return err
}

// AddBlock takes one block's weights, in order.
func (p *NomicPipeline) AddBlock(b NomicBlockData) error {
	if p.built {
		return fmt.Errorf("vk: the encoder is already on the card")
	}
	s := p.s
	var at nomicBlockAt
	var err error
	if at.qkv, err = p.keepLinear(b.QKV, 3*s.Dim, s.Dim); err != nil {
		return err
	}
	if at.o, err = p.keepLinear(b.O, s.Dim, s.Dim); err != nil {
		return err
	}
	if at.attn, err = p.keepNorm(b.AttnGain, b.AttnBias); err != nil {
		return err
	}
	if at.out, err = p.keepNorm(b.OutGain, b.OutBias); err != nil {
		return err
	}
	if b.Router == nil {
		if at.up, err = p.keepLinear(b.Up, s.FF, s.Dim); err != nil {
			return err
		}
		if at.dn, err = p.keepLinear(b.Down, s.Dim, s.FF); err != nil {
			return err
		}
	} else {
		if len(b.Router) != s.Experts*s.Dim {
			return fmt.Errorf("vk: a router of %d experts over %d is %d floats, given %d",
				s.Experts, s.Dim, s.Experts*s.Dim, len(b.Router))
		}
		if len(b.UpExps) != s.Experts || len(b.DownExps) != s.Experts {
			return fmt.Errorf("vk: a mixture of %d experts was given %d and %d matrices",
				s.Experts, len(b.UpExps), len(b.DownExps))
		}
		at.mixture = true
		at.router = uint32(len(p.par))
		p.par = append(p.par, b.Router...)
		for e := 0; e < s.Experts; e++ {
			up, err := p.keepLinear(b.UpExps[e], s.FF, s.Dim)
			if err != nil {
				return err
			}
			dn, err := p.keepLinear(b.DownExps[e], s.Dim, s.FF)
			if err != nil {
				return err
			}
			at.upE, at.dnE = append(at.upE, up), append(at.dnE, dn)
		}
		// The card's routing writes the grouped products' table itself, and
		// reads where each expert's matrices start from here.
		at.bases = uint32(len(p.par))
		for e := 0; e < s.Experts; e++ {
			p.par = append(p.par, math.Float32frombits(at.upE[e].base), math.Float32frombits(at.dnE[e].base))
		}
	}
	p.blocks = append(p.blocks, at)
	return nil
}

// Prepare uploads everything. The whole encoder is under a gigabyte in fp16,
// so it is resident or it is nothing.
func (p *NomicPipeline) Prepare() error {
	if p.built {
		return nil
	}
	var err error
	if p.parBuf, err = p.d.Upload(asBytes(p.par)); err != nil {
		return err
	}
	if p.geluBuf, err = p.d.Upload(asBytes(nn.GELUTableData())); err != nil {
		return err
	}
	if p.weights, err = p.d.Local(p.total, bufferUsageStorage|bufferUsageTransferDst); err != nil {
		return fmt.Errorf("vk: the card has no room for the encoder's %d MiB: %w", p.total>>20, err)
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
		// The host's copy is no longer needed, and for a file that was not
		// fp16 it is a copy the caller made.
		m.src = nil
	}
	p.built = true
	return nil
}

// Trace makes the next Encode keep three grids a block, under llama.cpp's
// names, which Waypoint then hands back. It submits a recording a block.
func (p *NomicPipeline) Trace() { p.trace = map[string][]float32{} }

// Waypoint is what the last traced Encode left under that name, or nil.
func (p *NomicPipeline) Waypoint(name string) []float32 { return p.trace[name] }

// Encode runs the blocks over one pass.
//
// x is each position's embedding with the sentence-A row added, not yet
// normed, Dim floats a position; pos is each position's place in its text and
// seg its text's start and length, two a position, the texts one after another.
// out receives what the last block wrote, Dim a position, and may be x.
func (p *NomicPipeline) Encode(x []float32, pos, seg []uint32, route NomicRouter, out []float32) error {
	if !p.built {
		return fmt.Errorf("vk: the encoder has not been prepared")
	}
	s := p.s
	n := len(x) / s.Dim
	if n == 0 || n*s.Dim != len(x) || len(pos) != n || len(seg) != 2*n || len(out) != len(x) {
		return fmt.Errorf("vk: an encoder pass of %d values wants whole positions of %d, a position each and two segment bounds each",
			len(x), s.Dim)
	}
	if err := p.scratchFor(n); err != nil {
		return err
	}
	sc := p.sc
	copy(sc.xin.Floats(), x)
	copy(sc.posin.Uints(), pos)
	copy(sc.segin.Uints(), seg)

	// The tiled attention's units: each text cut into stretches of queries.
	units := sc.unitin.Uints()[:0]
	for t := 0; t < n; {
		start, length := seg[2*t], seg[2*t+1]
		if int(start) != t || length == 0 || t+int(length) > n {
			return fmt.Errorf("vk: position %d says its text starts at %d and is %d long", t, start, length)
		}
		for q0 := uint32(0); q0 < length; q0 += uint32(max(p.queryTile, 1)) {
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

	pending = append(pending, func(r *Recorder) {
		if p.tl != nil {
			p.tl.Reset(r)
			p.stamp(r, "start")
		}
		r.Copy(sc.x, 0, sc.xin, n*s.Dim*4)
		r.Copy(sc.pos, 0, sc.posin, n*4)
		r.Copy(sc.seg, 0, sc.segin, 2*n*4)
		r.Copy(sc.unit, 0, sc.unitin, sc.units*16)
		r.Barrier()
		p.norm(r, sc.normE, p.emb, n)
		r.Barrier()
		p.stamp(r, "embed")
	})

	for i := range p.blocks {
		b := &p.blocks[i]
		pending = append(pending, func(r *Recorder) { p.recordAttention(r, b, n) })
		if !b.mixture {
			pending = append(pending, func(r *Recorder) {
				p.gemm(r, sc.gUp, sc.bUp, sc.sUp, sc.rUp, &b.up, n)
				r.Barrier()
				p.stamp(r, "up")
				p.act(r, sc.gelu, n*s.FF, 0)
				r.Barrier()
				p.stamp(r, "gelu")
				p.gemm(r, sc.gDn, sc.bDn, sc.sDn, sc.rTmp, &b.dn, n)
				r.Barrier()
				p.stamp(r, "down")
			})
		} else if p.trace == nil && n*s.Used <= nomicRouteSlots && s.Experts <= nomicRouteExperts {
			// The routing on the card: no round trip, and the pass stays one
			// submission. The tables are sized for the worst the counts could
			// be, and the entries the routing leaves empty cost a workgroup
			// that leaves at once.
			slots := n * s.Used
			widthUp, widthDn := 64, 64
			if p.big(s.Dim, ((s.FF+127)/128)*((slots+127)/128)) {
				widthUp = 128
			}
			if p.big(s.FF, ((s.Dim+127)/128)*((slots+127)/128)) {
				widthDn = 128
			}
			maxUp := (slots+widthUp-1)/widthUp + s.Experts
			maxDn := (slots+widthDn-1)/widthDn + s.Experts
			pending = append(pending, func(r *Recorder) {
				push := nomicRouterPush{dim: uint32(s.Dim), experts: uint32(s.Experts), at: b.router}
				r.Dispatch(sc.router, uint32(n), unsafe.Pointer(&push))
				r.Barrier()
				route := nomicRoutePush{n: uint32(n), experts: uint32(s.Experts), used: uint32(s.Used),
					bases: b.bases, widthUp: uint32(widthUp), widthDn: uint32(widthDn),
					maxUp: uint32(maxUp), maxDn: uint32(maxDn)}
				r.Dispatch(sc.route, 1, unsafe.Pointer(&route))
				r.Barrier()
				p.stamp(r, "route")
				up := nomicGemmPush{outputs: uint32(s.FF), inputs: uint32(s.Dim), bias: ^uint32(0), index: 0}
				p.grouped(r, sc.upE, sc.bUpE, sc.sUpE, sc.rUp, up, widthUp, maxUp, slots)
				r.Barrier()
				p.stamp(r, "expert up")
				p.act(r, sc.gelu, slots*s.FF, 0)
				r.Barrier()
				p.stamp(r, "gelu")
				dn := nomicGemmPush{outputs: uint32(s.Dim), inputs: uint32(s.FF), patches: uint32(maxUp),
					bias: ^uint32(0), index: 1}
				p.grouped(r, sc.dnE, sc.bDnE, sc.sDnE, sc.rAns, dn, widthDn, maxDn, slots)
				r.Barrier()
				p.stamp(r, "expert down")
				comb := nomicCombinePush{dim: uint32(s.Dim), used: uint32(s.Used), count: uint32(n * s.Dim)}
				r.Dispatch(sc.combine, uint32((n*s.Dim+255)/256), unsafe.Pointer(&comb))
				r.Barrier()
				p.stamp(r, "combine")
			})
		} else {
			pending = append(pending, func(r *Recorder) {
				push := nomicRouterPush{dim: uint32(s.Dim), experts: uint32(s.Experts), at: b.router}
				r.Dispatch(sc.router, uint32(n), unsafe.Pointer(&push))
				r.Barrier()
				r.Copy(sc.logitsBack, 0, sc.logits, n*s.Experts*4)
				p.stamp(r, "router")
			})
			if err := flush(); err != nil {
				return fmt.Errorf("vk: the encoder failed at block %d: %w", i, err)
			}
			idx, dest, wt, counts := route(i, append([]float32(nil), sc.logitsBack.Floats()[:n*s.Experts]...))
			if len(idx) != len(dest) || len(idx) > n*s.Used || len(wt) != n*s.Used || len(counts) != s.Experts {
				return fmt.Errorf("vk: the router answered %d positions, %d slots, %d weights and %d counts",
					len(idx), len(dest), len(wt), len(counts))
			}
			// A table per half of the expert, one entry per tile's worth of one
			// expert's choices: sixty-four or a hundred and twenty-eight,
			// whichever tile that half takes.
			entriesFor := func(width int) int {
				n := 0
				for _, c := range counts {
					n += (c + width - 1) / width
				}
				return n
			}
			bigUp := p.big(s.Dim, ((s.FF+127)/128)*entriesFor(128))
			bigDn := p.big(s.FF, ((s.Dim+127)/128)*entriesFor(128))
			table := sc.tablein.Uints()[:0]
			fill := func(width int) {
				at := 0
				for e, c := range counts {
					for c0 := 0; c0 < c; c0 += width {
						table = append(table, b.upE[e].base, b.dnE[e].base, uint32(at), uint32(c), uint32(c0), 0, 0, 0)
					}
					at += c
				}
			}
			widthUp, widthDn := 64, 64
			if bigUp {
				widthUp = 128
			}
			if bigDn {
				widthDn = 128
			}
			fill(widthUp)
			upEntries := len(table) / nomicTableEntry
			fill(widthDn)
			dnEntries := len(table)/nomicTableEntry - upEntries
			entries := upEntries + dnEntries
			copy(sc.idxin.Uints(), idx)
			copy(sc.destin.Uints(), dest)
			copy(sc.wtin.Floats(), wt)
			pending = append(pending, func(r *Recorder) {
				r.Copy(sc.idx, 0, sc.idxin, len(idx)*4)
				r.Copy(sc.dest, 0, sc.destin, len(dest)*4)
				r.Copy(sc.wt, 0, sc.wtin, len(wt)*4)
				r.Copy(sc.table, 0, sc.tablein, entries*nomicTableEntry*4)
				r.Barrier()
				p.stamp(r, "route upload")
				up := nomicGemmPush{outputs: uint32(s.FF), inputs: uint32(s.Dim), bias: ^uint32(0), index: 0}
				if bigUp {
					r.DispatchColumns(sc.bUpE, uint32((s.FF+127)/128), uint32(upEntries), unsafe.Pointer(&up))
				} else {
					r.DispatchColumns(sc.upE, uint32((s.FF+63)/64), uint32(upEntries), unsafe.Pointer(&up))
				}
				r.Barrier()
				p.stamp(r, "expert up")
				p.act(r, sc.gelu, len(idx)*s.FF, 0)
				r.Barrier()
				p.stamp(r, "gelu")
				dn := nomicGemmPush{outputs: uint32(s.Dim), inputs: uint32(s.FF), patches: uint32(upEntries),
					bias: ^uint32(0), index: 1}
				if bigDn {
					r.DispatchColumns(sc.bDnE, uint32((s.Dim+127)/128), uint32(dnEntries), unsafe.Pointer(&dn))
				} else {
					r.DispatchColumns(sc.dnE, uint32((s.Dim+63)/64), uint32(dnEntries), unsafe.Pointer(&dn))
				}
				r.Barrier()
				p.stamp(r, "expert down")
				comb := nomicCombinePush{dim: uint32(s.Dim), used: uint32(s.Used), count: uint32(n * s.Dim)}
				r.Dispatch(sc.combine, uint32((n*s.Dim+255)/256), unsafe.Pointer(&comb))
				r.Barrier()
				p.stamp(r, "combine")
			})
		}
		pending = append(pending, func(r *Recorder) {
			p.act(r, sc.addH, n*s.Dim, 1)
			r.Barrier()
			p.norm(r, sc.normX, b.out, n)
			r.Barrier()
			p.stamp(r, "add+norm")
			p.tap(r, 2, sc.x, n)
		})
		if p.trace != nil {
			if err := flush(); err != nil {
				return fmt.Errorf("vk: the encoder failed at block %d: %w", i, err)
			}
			all := sc.tap.Floats()
			for k, name := range nomicTaps {
				p.trace[fmt.Sprintf("%s-%d", name, i)] = append([]float32(nil), all[k*n*s.Dim:(k+1)*n*s.Dim]...)
			}
		}
	}

	pending = append(pending, func(r *Recorder) {
		r.Copy(sc.back, 0, sc.x, n*s.Dim*4)
	})
	if err := flush(); err != nil {
		return fmt.Errorf("vk: the encoder failed reading back: %w", err)
	}
	copy(out, sc.back.Floats()[:n*s.Dim])
	return nil
}

// recordAttention is a block's first half: the fused projection, the
// rotation, the attention, the output projection, the residual and the norm,
// which leaves ffn_inp in h.
func (p *NomicPipeline) recordAttention(r *Recorder, b *nomicBlockAt, n int) {
	s, sc := p.s, p.sc
	p.gemm(r, sc.gQKV, sc.bQKV, sc.sQKV, sc.rQKV, &b.qkv, n)
	r.Barrier()
	p.stamp(r, "qkv")
	rope := nomicRopePush{dim: uint32(s.Dim), heads: uint32(s.Heads), head: uint32(s.HeadDim), base: s.RoPEBase}
	r.Dispatch(sc.rope, uint32(n), unsafe.Pointer(&rope))
	r.Barrier()
	p.stamp(r, "rope")
	attn := nomicAttnPush{dim: uint32(s.Dim), head: uint32(s.HeadDim),
		scale: float32(1 / math.Sqrt(float64(s.HeadDim)))}
	if p.flashPipe != nil {
		r.DispatchColumns(sc.attn, uint32(sc.units), uint32(s.Heads), unsafe.Pointer(&attn))
	} else {
		r.DispatchColumns(sc.attn, uint32(n), uint32(s.Heads), unsafe.Pointer(&attn))
	}
	r.Barrier()
	p.stamp(r, "attention")
	p.gemm(r, sc.gO, sc.bO, sc.sO, sc.rTmp, &b.o, n)
	r.Barrier()
	p.stamp(r, "o")
	p.tap(r, 0, sc.tmp, n)
	p.act(r, sc.addX, n*s.Dim, 1)
	r.Barrier()
	p.norm(r, sc.normH, b.attn, n)
	r.Barrier()
	p.stamp(r, "add+norm")
	p.tap(r, 1, sc.h, n)
}

func (p *NomicPipeline) norm(r *Recorder, set *Set, at visionNormAt, n int) {
	push := visionNormPush{dim: uint32(p.s.Dim), gain: at.gain, bias: at.bias, eps: p.s.Eps}
	r.Dispatch(set, uint32(n), unsafe.Pointer(&push))
}

// gemm dispatches one product of a matrix against columns positions, on the
// large tile when the pass is wide enough to fill the card with them.
//
// A pass too narrow for the large tile takes the small one, and one too narrow
// to fill the card even with that is split along the shared dimension: split
// writes the slices' partial sums, and reduce adds them where small would have
// written.
func (p *NomicPipeline) gemm(r *Recorder, small, large, split, reduce *Set, m *visionMat, columns int) {
	push := nomicGemmPush{outputs: uint32(m.outputs), inputs: uint32(m.inputs), patches: uint32(columns),
		base: m.base, bias: m.bias}
	if tiles := ((m.outputs + 127) / 128) * ((columns + 127) / 128); large != nil && p.big(m.inputs, tiles) {
		r.Dispatch(large, uint32(tiles), unsafe.Pointer(&push))
		return
	}
	groups := ((m.outputs + 63) / 64) * ((columns + 63) / 64)
	if split != nil {
		if s, kper := p.split(groups, m.inputs, columns, m.outputs); s > 1 {
			push.slices, push.stride, push.kper = uint32(s), uint32(columns*m.outputs), uint32(kper)
			r.Dispatch(split, uint32(groups*s), unsafe.Pointer(&push))
			r.Barrier()
			p.reduce(r, reduce, columns, m.outputs, s, m.bias)
			return
		}
	}
	r.Dispatch(small, uint32(groups), unsafe.Pointer(&push))
}

// grouped dispatches one half of every expert off the table: on the large
// tile when the routing sized the table for it, and otherwise on the small
// one, split along the shared dimension when the choices are too few to fill
// the card. slots is how many choices there are, which is how many rows the
// half writes.
func (p *NomicPipeline) grouped(r *Recorder, small, large, split, reduce *Set, push nomicGemmPush, width, entries, slots int) {
	outputs, inputs := int(push.outputs), int(push.inputs)
	if width == 128 {
		r.DispatchColumns(large, uint32((outputs+127)/128), uint32(entries), unsafe.Pointer(&push))
		return
	}
	rowTiles := (outputs + 63) / 64
	if split != nil {
		// The table has room for the worst case; the tiles that do work are
		// about one per sixty-four choices.
		if s, kper := p.split(rowTiles*((slots+63)/64), inputs, slots, outputs); s > 1 {
			push.slices, push.stride, push.kper = uint32(s), uint32(slots*outputs), uint32(kper)
			r.DispatchColumns(split, uint32(rowTiles*s), uint32(entries), unsafe.Pointer(&push))
			r.Barrier()
			p.reduce(r, reduce, slots, outputs, s, ^uint32(0))
			return
		}
	}
	r.DispatchColumns(small, uint32(rowTiles), uint32(entries), unsafe.Pointer(&push))
}

func (p *NomicPipeline) act(r *Recorder, set *Set, count int, op uint32) {
	push := visionActPush{count: uint32(count), op: op}
	r.Dispatch(set, uint32((count+255)/256), unsafe.Pointer(&push))
}

// tap keeps one grid of a traced block in its slot of the tap buffer.
func (p *NomicPipeline) tap(r *Recorder, slot int, src *Buffer, n int) {
	if p.trace == nil {
		return
	}
	r.Barrier()
	r.Copy(p.sc.tap, slot*n*p.s.Dim*4, src, n*p.s.Dim*4)
}

// scratchFor makes the buffers a pass of n positions needs, or keeps the ones
// a pass at least as wide left behind.
func (p *NomicPipeline) scratchFor(n int) error {
	if p.sc != nil && p.sc.n >= n && (p.trace == nil || p.sc.tap != nil) {
		return nil
	}
	if p.sc != nil {
		p.sc.close()
		p.sc = nil
	}
	s := p.s
	size := max(n, 64)
	sc := &nomicScratch{n: size}
	for _, c := range []struct {
		b     **Buffer
		words int
	}{
		{&sc.x, size * s.Dim},
		{&sc.h, size * s.Dim},
		{&sc.qkv, size * 3 * s.Dim},
		{&sc.q, size * s.Dim},
		{&sc.k, size * s.Dim},
		{&sc.mixed, size * s.Dim},
		{&sc.tmp, size * s.Dim},
		{&sc.up, size * max(s.Used, 1) * s.FF},
		{&sc.ans, size * max(s.Used, 1) * s.Dim},
		{&sc.logits, size * max(s.Experts, 1)},
		{&sc.pos, size},
		{&sc.seg, 2 * size},
		{&sc.idx, size * max(s.Used, 1)},
		{&sc.dest, size * max(s.Used, 1)},
		{&sc.wt, size * max(s.Used, 1)},
		{&sc.unit, 4 * size},
		{&sc.table, 2 * nomicTableEntry * (size*max(s.Used, 1)/64 + s.Experts + 1)},
	} {
		v, err := p.d.Local(c.words*4, bufferUsageStorage|bufferUsageTransferSrc|bufferUsageTransferDst)
		if err != nil {
			sc.close()
			return fmt.Errorf("vk: the card has no room for a pass of %d positions: %w", size, err)
		}
		*c.b = v
	}
	var err error
	for _, c := range []struct {
		b     **Buffer
		bytes int
		back  bool
	}{
		{&sc.xin, size * s.Dim * 4, false},
		{&sc.posin, size * 4, false},
		{&sc.segin, 2 * size * 4, false},
		{&sc.idxin, size * max(s.Used, 1) * 4, false},
		{&sc.destin, size * max(s.Used, 1) * 4, false},
		{&sc.wtin, size * max(s.Used, 1) * 4, false},
		{&sc.unitin, 4 * size * 4, false},
		{&sc.tablein, 2 * nomicTableEntry * (size*max(s.Used, 1)/64 + s.Experts + 1) * 4, false},
		{&sc.back, size * s.Dim * 4, true},
		{&sc.logitsBack, size * max(s.Experts, 1) * 4, true},
	} {
		if c.back {
			*c.b, err = p.d.Readback(c.bytes, bufferUsageTransferDst)
		} else {
			*c.b, err = p.d.Host(c.bytes, bufferUsageTransferSrc)
		}
		if err != nil {
			sc.close()
			return err
		}
	}
	if p.trace != nil {
		if sc.tap, err = p.d.Readback(len(nomicTaps)*size*s.Dim*4, bufferUsageTransferDst); err != nil {
			sc.close()
			return err
		}
	}
	attnPipe, attnBuffers := p.attnPipe, []*Buffer{sc.q, sc.k, sc.qkv, sc.mixed, sc.seg}
	if p.flashPipe != nil {
		attnPipe, attnBuffers = p.flashPipe, []*Buffer{sc.q, sc.k, sc.qkv, sc.mixed, sc.unit}
	}
	for _, c := range []struct {
		set     **Set
		pipe    *Pipeline
		buffers []*Buffer
	}{
		{&sc.normE, p.normPipe, []*Buffer{sc.x, p.parBuf, sc.x}},
		{&sc.normH, p.normPipe, []*Buffer{sc.tmp, p.parBuf, sc.h}},
		{&sc.normX, p.normPipe, []*Buffer{sc.tmp, p.parBuf, sc.x}},
		{&sc.gQKV, p.gemmPipe, []*Buffer{p.weights, sc.x, p.parBuf, sc.qkv, sc.idx, sc.table}},
		{&sc.gO, p.gemmPipe, []*Buffer{p.weights, sc.mixed, p.parBuf, sc.tmp, sc.idx, sc.table}},
		{&sc.gUp, p.gemmPipe, []*Buffer{p.weights, sc.h, p.parBuf, sc.up, sc.idx, sc.table}},
		{&sc.gDn, p.gemmPipe, []*Buffer{p.weights, sc.up, p.parBuf, sc.tmp, sc.idx, sc.table}},
		{&sc.upE, p.groupPipe, []*Buffer{p.weights, sc.h, p.parBuf, sc.up, sc.idx, sc.table}},
		{&sc.dnE, p.groupPipe, []*Buffer{p.weights, sc.up, p.parBuf, sc.ans, sc.dest, sc.table}},
		{&sc.combine, p.combinePipe, []*Buffer{sc.ans, sc.wt, sc.tmp}},
		{&sc.route, p.routePipe, []*Buffer{sc.logits, p.parBuf, sc.idx, sc.dest, sc.wt, sc.table}},
		{&sc.rope, p.ropePipe, []*Buffer{sc.qkv, sc.pos, sc.q, sc.k}},
		{&sc.attn, attnPipe, attnBuffers},
		{&sc.router, p.routerPipe, []*Buffer{p.parBuf, sc.h, sc.logits}},
		{&sc.addX, p.actPipe, []*Buffer{sc.tmp, sc.x, p.geluBuf}},
		{&sc.addH, p.actPipe, []*Buffer{sc.tmp, sc.h, p.geluBuf}},
		{&sc.gelu, p.actPipe, []*Buffer{sc.up, sc.up, p.geluBuf}},
	} {
		set, err := c.pipe.NewSet(c.buffers)
		if err != nil {
			sc.close()
			return err
		}
		*c.set = set
	}
	if p.bigPipe != nil {
		if sc.partial, err = p.d.Local(nomicPartialFloats*4, bufferUsageStorage); err != nil {
			sc.close()
			return err
		}
		for _, c := range []struct {
			set     **Set
			pipe    *Pipeline
			buffers []*Buffer
		}{
			{&sc.sQKV, p.gemmPipe, []*Buffer{p.weights, sc.x, p.parBuf, sc.partial, sc.idx, sc.table}},
			{&sc.sO, p.gemmPipe, []*Buffer{p.weights, sc.mixed, p.parBuf, sc.partial, sc.idx, sc.table}},
			{&sc.sUp, p.gemmPipe, []*Buffer{p.weights, sc.h, p.parBuf, sc.partial, sc.idx, sc.table}},
			{&sc.sDn, p.gemmPipe, []*Buffer{p.weights, sc.up, p.parBuf, sc.partial, sc.idx, sc.table}},
			{&sc.sUpE, p.groupPipe, []*Buffer{p.weights, sc.h, p.parBuf, sc.partial, sc.idx, sc.table}},
			{&sc.sDnE, p.groupPipe, []*Buffer{p.weights, sc.up, p.parBuf, sc.partial, sc.dest, sc.table}},
			{&sc.rQKV, p.reducePipe, []*Buffer{sc.partial, p.parBuf, sc.qkv}},
			{&sc.rTmp, p.reducePipe, []*Buffer{sc.partial, p.parBuf, sc.tmp}},
			{&sc.rUp, p.reducePipe, []*Buffer{sc.partial, p.parBuf, sc.up}},
			{&sc.rAns, p.reducePipe, []*Buffer{sc.partial, p.parBuf, sc.ans}},
			{&sc.bQKV, p.bigPipe, []*Buffer{p.weights, sc.x, p.parBuf, sc.qkv, sc.idx, sc.table}},
			{&sc.bO, p.bigPipe, []*Buffer{p.weights, sc.mixed, p.parBuf, sc.tmp, sc.idx, sc.table}},
			{&sc.bUp, p.bigPipe, []*Buffer{p.weights, sc.h, p.parBuf, sc.up, sc.idx, sc.table}},
			{&sc.bDn, p.bigPipe, []*Buffer{p.weights, sc.up, p.parBuf, sc.tmp, sc.idx, sc.table}},
			{&sc.bUpE, p.bigGroupPipe, []*Buffer{p.weights, sc.h, p.parBuf, sc.up, sc.idx, sc.table}},
			{&sc.bDnE, p.bigGroupPipe, []*Buffer{p.weights, sc.up, p.parBuf, sc.ans, sc.dest, sc.table}},
		} {
			set, err := c.pipe.NewSet(c.buffers)
			if err != nil {
				sc.close()
				return err
			}
			*c.set = set
		}
	}
	p.sc = sc
	return nil
}

func (sc *nomicScratch) close() {
	for _, s := range []**Set{&sc.normE, &sc.normH, &sc.normX, &sc.gQKV, &sc.gO, &sc.gUp, &sc.gDn,
		&sc.upE, &sc.dnE, &sc.combine, &sc.rope, &sc.attn, &sc.router, &sc.addX, &sc.addH, &sc.gelu,
		&sc.bQKV, &sc.bO, &sc.bUp, &sc.bDn, &sc.bUpE, &sc.bDnE, &sc.route,
		&sc.sQKV, &sc.sO, &sc.sUp, &sc.sDn, &sc.sUpE, &sc.sDnE, &sc.rQKV, &sc.rTmp, &sc.rUp, &sc.rAns} {
		if *s != nil {
			(*s).Close()
			*s = nil
		}
	}
	for _, b := range []**Buffer{&sc.x, &sc.h, &sc.qkv, &sc.q, &sc.k, &sc.mixed, &sc.tmp, &sc.up,
		&sc.ans, &sc.logits, &sc.pos, &sc.seg, &sc.idx, &sc.dest, &sc.wt, &sc.unit, &sc.table,
		&sc.xin, &sc.posin, &sc.segin, &sc.idxin, &sc.destin, &sc.wtin, &sc.unitin, &sc.tablein,
		&sc.back, &sc.logitsBack, &sc.tap, &sc.partial} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
}

// Close releases everything the encoder holds on the device.
func (p *NomicPipeline) Close() {
	if p.sc != nil {
		p.sc.close()
		p.sc = nil
	}
	if p.tl != nil {
		p.tl.Close()
		p.tl = nil
	}
	for _, pipe := range []**Pipeline{&p.normPipe, &p.gemmPipe, &p.groupPipe, &p.combinePipe,
		&p.ropePipe, &p.attnPipe, &p.flashPipe, &p.actPipe, &p.routerPipe, &p.bigPipe, &p.bigGroupPipe, &p.routePipe, &p.reducePipe} {
		if *pipe != nil {
			(*pipe).Close()
			*pipe = nil
		}
	}
	for _, b := range []**Buffer{&p.parBuf, &p.geluBuf, &p.weights} {
		if *b != nil {
			(*b).Close()
			*b = nil
		}
	}
	p.built = false
}
