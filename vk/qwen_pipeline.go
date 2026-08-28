package vk

// Every block of Qwen3.5 on the card, one submission to a token.
//
// The model is two kinds of block interleaved three to one: forty-eight gated
// delta nets, whose state is a 128x128 matrix a head rather than a growing
// cache, and sixteen full attentions that keep an ordinary one. Both end in the
// same feed forward, so what differs between them is the mixer alone and
// everything around it is written once.
//
// The reason it is one submission and not one a block is vk/stack.go's: a
// submission costs sixty-three microseconds whatever is in it, and a card
// handed a hundred microseconds of work and then left alone drops to half its
// clocks. Sixty-four blocks submitted apart would be four milliseconds of
// nothing before any arithmetic happened.

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q40.comp -o shaders/matvec_q40.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q41.comp -o shaders/matvec_q41.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/swiglu_act.comp -o shaders/swiglu_act.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/qwen_attn_prep.comp -o shaders/qwen_attn_prep.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/qwen_attn_gqa.comp -o shaders/qwen_attn_gqa.spv

//go:embed shaders/matvec_q40.spv
var matvecQ40SPIRV []byte

//go:embed shaders/matvec_q41.spv
var matvecQ41SPIRV []byte

//go:embed shaders/matvec_f32.spv
var matvecF32SPIRV []byte

//go:embed shaders/matvec_q80.spv
var matvecQ80SPIRV []byte

// The same five kernels built for a pass of two, four and eight columns.
//
// Two is what a draft verified beside the token that drafted it needs, and a
// pass of two costs 1.056 of a pass of one: the weights are read once either
// way, and a token is bounded by reading them. Reading a prompt is the same
// bargain taken further — eight columns for one reading of the model — and
// eight is where a mat-vec stops, because past it the accumulator a thread
// carries a column in stops fitting in registers and the answer is a tiled
// product rather than a wider mat-vec. vk/matmul.go's smallColumns says the
// same number for the same reason.
//
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec2.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q40.comp -o shaders/matvec_q40_2.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q41.comp -o shaders/matvec_q41_2.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q5k.comp -o shaders/matvec_q5k_2.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32_2.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec4.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q40.comp -o shaders/matvec_q40_4.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q41.comp -o shaders/matvec_q41_4.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q5k.comp -o shaders/matvec_q5k_4.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32_4.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q40.comp -o shaders/matvec_q40_8.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q41.comp -o shaders/matvec_q41_8.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q5k.comp -o shaders/matvec_q5k_8.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32_8.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_qwen16.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q40.comp -o shaders/matvec_q40_16.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q41.comp -o shaders/matvec_q41_16.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q5k.comp -o shaders/matvec_q5k_16.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32_16.spv

//go:embed shaders/matvec2.spv
var matvec2SPIRV []byte

//go:embed shaders/matvec_q40_2.spv
var matvecQ40_2SPIRV []byte

//go:embed shaders/matvec_q41_2.spv
var matvecQ41_2SPIRV []byte

//go:embed shaders/matvec_q5k_2.spv
var matvecQ5K_2SPIRV []byte

//go:embed shaders/matvec_f32_2.spv
var matvecF32_2SPIRV []byte

//go:embed shaders/matvec4.spv
var matvec4SPIRV []byte

// matvec8.spv is vk/mixture.go's, built from the same source at eight columns.
// The delta net's largest two projections read it: its q+k+v and its gate are
// the widest matrices in the block, and eight columns is where a mat-vec ends.
//
//go:embed shaders/matvec8.spv
var matvecQwen8SPIRV []byte

//go:embed shaders/matvec_q40_4.spv
var matvecQ40_4SPIRV []byte

//go:embed shaders/matvec_q41_4.spv
var matvecQ41_4SPIRV []byte

//go:embed shaders/matvec_q5k_4.spv
var matvecQ5K_4SPIRV []byte

//go:embed shaders/matvec_f32_4.spv
var matvecF32_4SPIRV []byte

//go:embed shaders/matvec_q40_8.spv
var matvecQ40_8SPIRV []byte

//go:embed shaders/matvec_q41_8.spv
var matvecQ41_8SPIRV []byte

//go:embed shaders/matvec_q5k_8.spv
var matvecQ5K_8SPIRV []byte

//go:embed shaders/matvec_f32_8.spv
var matvecF32_8SPIRV []byte

//go:embed shaders/matvec_qwen16.spv
var matvecQwen16SPIRV []byte

//go:embed shaders/matvec_q40_16.spv
var matvecQ40_16SPIRV []byte

//go:embed shaders/matvec_q41_16.spv
var matvecQ41_16SPIRV []byte

//go:embed shaders/matvec_q5k_16.spv
var matvecQ5K_16SPIRV []byte

//go:embed shaders/matvec_f32_16.spv
var matvecF32_16SPIRV []byte

// qwenWide is the widest pass any of these binaries answers. Everything
// per-column is allocated for it.
const qwenWide = 16

// qwenWidths are the widths a pass may take, largest first. A run of tokens is
// cut into passes of these: a hundred positions is twelve of eight and one of
// four, and the remainder never falls back to one column at a time unless it
// is one column.
var qwenWidths = [...]int{16, 8, 4, 2, 1}

//go:embed shaders/swiglu_act.spv
var swigluActSPIRV []byte

//go:embed shaders/qwen_attn_prep.spv
var qwenAttnPrepSPIRV []byte

//go:embed shaders/qwen_attn_gqa.spv
var qwenAttnGQASPIRV []byte

// norm.spv built with a staging array wide enough for a 5120-wide hidden
// state. The default binary stages 4096 and would overrun it.
//
//go:embed shaders/norm_wide.spv
var normWideSPIRV []byte

// qwenMaxContext is the longest run shaders/qwen_attn_gqa.comp holds scores
// for, which is its MAXCTX. A model asked for more than this cannot use the
// pipeline at all, and says so rather than answering out of a shorter array.
const qwenMaxContext = 8192

// matvecRows is how many outputs one workgroup of the mat-vec kernels answers.
// All five of them — Q4_0 against Q8_0, and Q4_0, Q4_1, Q5_K and F32 against
// floats — put 128 threads on sixteen rows with eight lanes to a row.
const matvecRows = 16

func matvecGroups(rows int) uint32 { return uint32((rows + matvecRows - 1) / matvecRows) }

type swigluPush struct {
	N uint32
}

type attnPrepPush struct {
	MaxContext uint32
	Heads      uint32
	KVHeads    uint32
	RoPEBase   float32
	Eps        float32
	RoPEDims   uint32
}

type attnGQAPush struct {
	MaxContext uint32
	HeadsPerKV uint32
	Heads      uint32
	Scale      float32
}

// QwenShape is the geometry every block of one model shares. It is passed once
// rather than rediscovered per block because nothing in Qwen3.5 varies from
// block to block except which of the two mixers a block has.
type QwenShape struct {
	Dim        int // 5120
	FFN        int // 17408
	MaxContext int

	Heads    int // 24 query heads
	KVHeads  int // 4
	HeadDim  int // 256
	RoPEDims int // 64
	RoPEBase float32

	ConvDim   int // 10240, the delta net's q+k+v channels
	Inner     int // 6144, its value width
	Rank      int // 48 value heads
	StateSize int // 128
	Groups    int // 16 key heads

	Eps float32
}

func (s QwenShape) qDim() int     { return s.Heads * s.HeadDim }
func (s QwenShape) qFullDim() int { return s.qDim() * 2 }
func (s QwenShape) kvDim() int    { return s.KVHeads * s.HeadDim }

type QwenSSMData struct {
	WQKV     []byte
	WGate    []byte
	WAlpha   []byte
	WBeta    []byte
	WOut     []byte
	OutIsQ5K bool
	OutIsF32 bool

	ConvWeight []float32
	SSMA       []float32
	SSMDtBias  []float32
	SSMNorm    []float32
}

type QwenAttnData struct {
	WQ    []byte
	WK    []byte
	WV    []byte
	WO    []byte
	QNorm []float32
	KNorm []float32
}

type QwenFFNData struct {
	Gate     []byte
	Up       []byte
	Down     []byte
	DownQ4_1 bool
}

type qwenSSMBlock struct {
	wQKV, wGate, wAlpha, wBeta, wOut *Buffer

	convWeight, convState, convOut *Buffer
	shadowState, shadowConv        *Buffer
	qkvBuf, gateZBuf               *Buffer
	alphaBuf, betaBuf              *Buffer
	dtBias, ssmA, ssmNorm          *Buffer
	ssmState, ySSM, outBuf         *Buffer

	setQKV, setGate, setAlpha, setBeta *Set
	setConv, setScan, setOut           *Set
}

type qwenAttnBlock struct {
	wQ, wK, wV, wO *Buffer
	qNorm, kNorm   *Buffer

	qIn, kIn, vIn  *Buffer
	qOut, attnOut  *Buffer
	kCache, vCache *Buffer
	outBuf         *Buffer

	setQ, setK, setV *Set
	setPrep, setGQA  *Set
	setO             *Set
}

type qwenFFNBlock struct {
	wGate, wUp, wDown       *Buffer
	setGate, setUp, setDown *Set
}

type QwenPipeline struct {
	d     *Device
	shape QwenShape

	pipeNorm     *Pipeline
	pipeMatvec   *Pipeline // Q4_0 against the Q8_0 activation
	pipeMatQ40   *Pipeline // Q4_0 against floats
	pipeMatQ41   *Pipeline
	pipeMatQ5K   *Pipeline
	pipeMatF32   *Pipeline
	pipeSwiglu   *Pipeline
	pipeConv     *Pipeline
	pipeScan     *Pipeline
	pipeAttnPrep *Pipeline
	pipeAttnGQA  *Pipeline
	pipeMatQ80   *Pipeline // the prediction block's front projection

	// The stream, which lives in device memory. xin and stage are its host
	// ends: one copy in at the head of a pass and one out at the foot, rather
	// than sixty-four blocks reading the hidden state across the bus.
	xin    *Buffer
	xs     *Buffer
	stage  *Buffer
	hidden *Buffer // the same state without the output norm, for the MTP block
	probe  *Buffer // a wide readback the tests borrow; nil until one asks

	// The position, which is the only thing about a pass that changes from one
	// token to the next. It is a buffer so that the recording does not have to
	// be laid down again for it: see shaders/qwen_attn_prep.comp.
	posIn  *Buffer
	posBuf *Buffer
	pass   map[int]*Program

	resid    *Buffer
	normed   *Buffer
	normedQ  *Buffer
	normedS  *Buffer
	ffnNorm  *Buffer
	ffnNormQ *Buffer
	ffnNormS *Buffer
	none     *Buffer

	gateBuf *Buffer
	upBuf   *Buffer
	actBuf  *Buffer
	ffnOut  *Buffer
	setAct  *Set

	attnNorms []*Buffer
	ffnNorms  []*Buffer
	outNorm   *Buffer

	setAttnNorms []*Set
	setFFNNorms  []*Set
	setFinalNorm *Set

	// The probes need a norm that reads the stream rather than closing the
	// block before it, and one residual add with no norm on top. Neither is on
	// the pass; both exist so a divergence can be stopped anywhere.
	setProbeNorms []*Set
	setResidOnly  *Set

	isSSM      []bool
	ssmBlocks  map[int]*qwenSSMBlock
	attnBlocks map[int]*qwenAttnBlock
	ffnBlocks  []*qwenFFNBlock
	mtp        *qwenMTPBlock

	// The pass that puts the delta nets back when a draft is refused.
	restoreProg *Program

	owned []*Buffer
}

func NewQwenPipeline(d *Device, shape QwenShape) (*QwenPipeline, error) {
	if shape.MaxContext > qwenMaxContext {
		return nil, fmt.Errorf("vk: qwen pipeline holds %d positions of scores, not %d", qwenMaxContext, shape.MaxContext)
	}
	p := &QwenPipeline{
		d:          d,
		shape:      shape,
		ssmBlocks:  make(map[int]*qwenSSMBlock),
		attnBlocks: make(map[int]*qwenAttnBlock),
	}
	var err error

	type build struct {
		into   **Pipeline
		spirv  []byte
		binds  int
		pushSz uintptr
	}
	for _, b := range []build{
		{&p.pipeNorm, normWideSPIRV, 8, unsafe.Sizeof(normPush{})},
		{&p.pipeMatvec, matvecSPIRV, 4, unsafe.Sizeof(moePush{})},
		{&p.pipeMatQ40, matvecQ40SPIRV, 3, unsafe.Sizeof(matvecKPush{})},
		{&p.pipeMatQ41, matvecQ41SPIRV, 3, unsafe.Sizeof(matvecKPush{})},
		{&p.pipeMatQ5K, matvecQ5KSPIRV, 3, unsafe.Sizeof(matvecKPush{})},
		{&p.pipeMatF32, matvecF32SPIRV, 3, unsafe.Sizeof(matvecKPush{})},
		{&p.pipeSwiglu, swigluActSPIRV, 3, unsafe.Sizeof(swigluPush{})},
		{&p.pipeConv, ssmConv1dSPIRV, 5, unsafe.Sizeof(ssmConvPush{})},
		{&p.pipeScan, ssmScanSPIRV, 10, unsafe.Sizeof(ssmScanPush{})},
		{&p.pipeAttnPrep, qwenAttnPrepSPIRV, 9, unsafe.Sizeof(attnPrepPush{})},
		{&p.pipeAttnGQA, qwenAttnGQASPIRV, 6, unsafe.Sizeof(attnGQAPush{})},
	} {
		if *b.into, err = d.NewPipeline(b.spirv, b.binds, uint32(b.pushSz)); err != nil {
			return nil, err
		}
	}
	// The wide binaries share their pipelines' layouts, so the sets made for
	// the one-column form reach them without being made again. Every one of
	// them has to exist at every width qwenWidths names: a pass asks for its
	// width by name and there is no falling back to a narrower binary.
	for _, w := range []struct {
		pipe    *Pipeline
		columns int
		spirv   []byte
	}{
		{p.pipeMatvec, 2, matvec2SPIRV}, {p.pipeMatvec, 4, matvec4SPIRV}, {p.pipeMatvec, 8, matvecQwen8SPIRV}, {p.pipeMatvec, 16, matvecQwen16SPIRV},
		{p.pipeMatQ40, 2, matvecQ40_2SPIRV}, {p.pipeMatQ40, 4, matvecQ40_4SPIRV}, {p.pipeMatQ40, 8, matvecQ40_8SPIRV}, {p.pipeMatQ40, 16, matvecQ40_16SPIRV},
		{p.pipeMatQ41, 2, matvecQ41_2SPIRV}, {p.pipeMatQ41, 4, matvecQ41_4SPIRV}, {p.pipeMatQ41, 8, matvecQ41_8SPIRV}, {p.pipeMatQ41, 16, matvecQ41_16SPIRV},
		{p.pipeMatQ5K, 2, matvecQ5K_2SPIRV}, {p.pipeMatQ5K, 4, matvecQ5K_4SPIRV}, {p.pipeMatQ5K, 8, matvecQ5K_8SPIRV}, {p.pipeMatQ5K, 16, matvecQ5K_16SPIRV},
		{p.pipeMatF32, 2, matvecF32_2SPIRV}, {p.pipeMatF32, 4, matvecF32_4SPIRV}, {p.pipeMatF32, 8, matvecF32_8SPIRV}, {p.pipeMatF32, 16, matvecF32_16SPIRV},
	} {
		if err := w.pipe.Wide(w.columns, w.spirv); err != nil {
			return nil, err
		}
	}

	dim := shape.Dim * qwenWide
	if p.xin, err = d.Host(dim*4, bufferUsageStorage|bufferUsageTransferSrc); err != nil {
		return nil, err
	}
	if p.stage, err = d.Readback(dim*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if p.hidden, err = d.Readback(dim*4, bufferUsageStorage); err != nil {
		return nil, err
	}
	if p.posIn, err = d.Host(64, bufferUsageStorage|bufferUsageTransferSrc); err != nil {
		return nil, err
	}
	if p.posBuf, err = d.Local(64, bufferUsageStorage); err != nil {
		return nil, err
	}
	for _, into := range []**Buffer{&p.xs, &p.resid, &p.normed, &p.normedQ, &p.normedS, &p.ffnNorm, &p.ffnNormQ, &p.ffnNormS, &p.ffnOut, &p.none} {
		if *into, err = p.local(dim * 4); err != nil {
			return nil, err
		}
	}
	for _, into := range []**Buffer{&p.gateBuf, &p.upBuf, &p.actBuf} {
		if *into, err = p.local(shape.FFN * qwenWide * 4); err != nil {
			return nil, err
		}
	}
	if p.setAct, err = p.pipeSwiglu.NewSet([]*Buffer{p.gateBuf, p.upBuf, p.actBuf}); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *QwenPipeline) local(bytes int) (*Buffer, error) {
	b, err := p.d.Local(bytes, bufferUsageStorage)
	if err != nil {
		return nil, err
	}
	p.owned = append(p.owned, b)
	return b, nil
}

// uploadQ4_0 and uploadQ4_1 lay a matrix out the way the mat-vec kernels read
// it: every scale of a row before any of its nibbles, rather than the file's
// interleaving of one scale and sixteen bytes at a time. Uploading the file's
// bytes straight through gives a kernel that reads a nibble where a scale is
// and answers infinities — which is what it did.
func (p *QwenPipeline) uploadQ4_0(data []byte, rows, cols int) (*Buffer, error) {
	return p.upload(splitQ4_0(data, rows, cols))
}

func (p *QwenPipeline) uploadQ4_1(data []byte, rows, cols int) (*Buffer, error) {
	return p.upload(splitQ4_1(data, rows, cols))
}

func (p *QwenPipeline) upload(data []byte) (*Buffer, error) {
	b, err := p.d.Upload(data)
	if err != nil {
		return nil, err
	}
	p.owned = append(p.owned, b)
	return b, nil
}

func (p *QwenPipeline) AddSSMBlock(i int, d QwenSSMData) error {
	s := p.shape
	b := &qwenSSMBlock{}
	var err error

	if b.wQKV, err = p.uploadQ4_0(d.WQKV, s.ConvDim, s.Dim); err != nil {
		return err
	}
	if b.wGate, err = p.uploadQ4_0(d.WGate, s.Inner, s.Dim); err != nil {
		return err
	}
	switch {
	case d.OutIsQ5K, d.OutIsF32:
		b.wOut, err = p.upload(d.WOut)
	default:
		b.wOut, err = p.uploadQ4_1(d.WOut, s.Dim, s.Inner)
	}
	if err != nil {
		return err
	}
	for _, u := range []struct {
		into **Buffer
		data []byte
	}{
		{&b.wAlpha, d.WAlpha}, {&b.wBeta, d.WBeta},
		{&b.convWeight, asBytes(d.ConvWeight)},
		{&b.dtBias, asBytes(d.SSMDtBias)}, {&b.ssmA, asBytes(d.SSMA)},
		{&b.ssmNorm, asBytes(d.SSMNorm)},
		{&b.convState, make([]byte, (s.ConvDim*4)*3)},
		{&b.ssmState, make([]byte, s.Rank*s.StateSize*s.StateSize*4)},
		{&b.shadowConv, make([]byte, (s.ConvDim*4)*3)},
		{&b.shadowState, make([]byte, s.Rank*s.StateSize*s.StateSize*4)},
	} {
		if *u.into, err = p.upload(u.data); err != nil {
			return err
		}
	}
	for _, l := range []struct {
		into  **Buffer
		bytes int
	}{
		{&b.convOut, s.ConvDim * qwenWide * 4}, {&b.qkvBuf, s.ConvDim * qwenWide * 4},
		{&b.gateZBuf, s.Inner * qwenWide * 4},
		{&b.alphaBuf, s.Rank * qwenWide * 4}, {&b.betaBuf, s.Rank * qwenWide * 4},
		{&b.ySSM, s.Inner * qwenWide * 4}, {&b.outBuf, s.Dim * qwenWide * 4},
	} {
		if *l.into, err = p.local(l.bytes); err != nil {
			return err
		}
	}

	if b.setQKV, err = p.pipeMatvec.NewSet([]*Buffer{b.wQKV, p.normedQ, p.normedS, b.qkvBuf}); err != nil {
		return err
	}
	if b.setGate, err = p.pipeMatvec.NewSet([]*Buffer{b.wGate, p.normedQ, p.normedS, b.gateZBuf}); err != nil {
		return err
	}
	if b.setAlpha, err = p.pipeMatF32.NewSet([]*Buffer{b.wAlpha, p.normed, b.alphaBuf}); err != nil {
		return err
	}
	if b.setBeta, err = p.pipeMatF32.NewSet([]*Buffer{b.wBeta, p.normed, b.betaBuf}); err != nil {
		return err
	}
	if b.setConv, err = p.pipeConv.NewSet([]*Buffer{b.convWeight, b.qkvBuf, b.convState, b.convOut, b.shadowConv}); err != nil {
		return err
	}
	if b.setScan, err = p.pipeScan.NewSet([]*Buffer{b.convOut, b.gateZBuf, b.alphaBuf, b.dtBias, b.ssmA, b.betaBuf, b.ssmNorm, b.ssmState, b.ySSM, b.shadowState}); err != nil {
		return err
	}
	outPipe := p.pipeMatQ41
	switch {
	case d.OutIsF32:
		outPipe = p.pipeMatF32
	case d.OutIsQ5K:
		outPipe = p.pipeMatQ5K
	}
	if b.setOut, err = outPipe.NewSet([]*Buffer{b.wOut, b.ySSM, b.outBuf}); err != nil {
		return err
	}

	p.ssmBlocks[i] = b
	return nil
}

func (p *QwenPipeline) AddAttnBlock(i int, d QwenAttnData) error {
	b, err := p.newAttnBlock(d)
	if err != nil {
		return err
	}
	p.attnBlocks[i] = b
	return nil
}

func (p *QwenPipeline) newAttnBlock(d QwenAttnData) (*qwenAttnBlock, error) {
	s := p.shape
	b := &qwenAttnBlock{}
	var err error

	for _, u := range []struct {
		into       **Buffer
		data       []byte
		rows, cols int
	}{
		{&b.wQ, d.WQ, s.qFullDim(), s.Dim},
		{&b.wK, d.WK, s.kvDim(), s.Dim},
		{&b.wV, d.WV, s.kvDim(), s.Dim},
		{&b.wO, d.WO, s.Dim, s.qDim()},
	} {
		if *u.into, err = p.uploadQ4_0(u.data, u.rows, u.cols); err != nil {
			return nil, err
		}
	}
	for _, u := range []struct {
		into **Buffer
		data []byte
	}{
		{&b.qNorm, asBytes(d.QNorm)}, {&b.kNorm, asBytes(d.KNorm)},
		{&b.kCache, make([]byte, s.KVHeads*s.MaxContext*s.HeadDim*4)},
		{&b.vCache, make([]byte, s.KVHeads*s.MaxContext*s.HeadDim*4)},
	} {
		if *u.into, err = p.upload(u.data); err != nil {
			return nil, err
		}
	}
	for _, l := range []struct {
		into  **Buffer
		bytes int
	}{
		{&b.qIn, s.qFullDim() * qwenWide * 4},
		{&b.kIn, s.kvDim() * qwenWide * 4}, {&b.vIn, s.kvDim() * qwenWide * 4},
		{&b.qOut, s.qDim() * qwenWide * 4}, {&b.attnOut, s.qDim() * qwenWide * 4},
		{&b.outBuf, s.Dim * qwenWide * 4},
	} {
		if *l.into, err = p.local(l.bytes); err != nil {
			return nil, err
		}
	}

	if b.setQ, err = p.pipeMatvec.NewSet([]*Buffer{b.wQ, p.normedQ, p.normedS, b.qIn}); err != nil {
		return nil, err
	}
	if b.setK, err = p.pipeMatvec.NewSet([]*Buffer{b.wK, p.normedQ, p.normedS, b.kIn}); err != nil {
		return nil, err
	}
	if b.setV, err = p.pipeMatvec.NewSet([]*Buffer{b.wV, p.normedQ, p.normedS, b.vIn}); err != nil {
		return nil, err
	}
	if b.setPrep, err = p.pipeAttnPrep.NewSet([]*Buffer{b.qIn, b.kIn, b.vIn, b.qNorm, b.kNorm, b.qOut, b.kCache, b.vCache, p.posBuf}); err != nil {
		return nil, err
	}
	if b.setGQA, err = p.pipeAttnGQA.NewSet([]*Buffer{b.qOut, b.kCache, b.vCache, b.qIn, b.attnOut, p.posBuf}); err != nil {
		return nil, err
	}
	if b.setO, err = p.pipeMatQ40.NewSet([]*Buffer{b.wO, b.attnOut, b.outBuf}); err != nil {
		return nil, err
	}

	return b, nil
}

func (p *QwenPipeline) AddFFNBlock(d QwenFFNData) error {
	b, err := p.newFFNBlock(d)
	if err != nil {
		return err
	}
	p.ffnBlocks = append(p.ffnBlocks, b)
	return nil
}

func (p *QwenPipeline) newFFNBlock(d QwenFFNData) (*qwenFFNBlock, error) {
	s := p.shape
	b := &qwenFFNBlock{}
	var err error
	if b.wGate, err = p.uploadQ4_0(d.Gate, s.FFN, s.Dim); err != nil {
		return nil, err
	}
	if b.wUp, err = p.uploadQ4_0(d.Up, s.FFN, s.Dim); err != nil {
		return nil, err
	}
	if d.DownQ4_1 {
		b.wDown, err = p.uploadQ4_1(d.Down, s.Dim, s.FFN)
	} else {
		b.wDown, err = p.uploadQ4_0(d.Down, s.Dim, s.FFN)
	}
	if err != nil {
		return nil, err
	}

	if b.setGate, err = p.pipeMatvec.NewSet([]*Buffer{b.wGate, p.ffnNormQ, p.ffnNormS, p.gateBuf}); err != nil {
		return nil, err
	}
	if b.setUp, err = p.pipeMatvec.NewSet([]*Buffer{b.wUp, p.ffnNormQ, p.ffnNormS, p.upBuf}); err != nil {
		return nil, err
	}
	down := p.pipeMatQ40
	if d.DownQ4_1 {
		down = p.pipeMatQ41
	}
	if b.setDown, err = down.NewSet([]*Buffer{b.wDown, p.actBuf, p.ffnOut}); err != nil {
		return nil, err
	}

	return b, nil
}

// SetNorms binds the gain vectors, and with them every kernel that reads the
// stream. It has to come after the blocks, because the residual norm of a
// block binds that block's mixer output.
func (p *QwenPipeline) SetNorms(attnNorms, ffnNorms [][]float32, outNorm []float32, isSSM []bool) error {
	p.isSSM = isSSM
	var err error

	// A block's input norm also closes the block before it: the residual add
	// that used to be a dispatch of its own is the norm kernel's ADD|SUM, which
	// it was already able to do. That is one dispatch and one barrier fewer per
	// block, and both of them cost more than the arithmetic they carried — the
	// norm runs in a single workgroup, so a dispatch of it is latency and not
	// work. Only the first block has nothing to close, and reads the stream
	// straight.
	for i, a := range attnNorms {
		buf, err := p.upload(asBytes(a))
		if err != nil {
			return err
		}
		p.attnNorms = append(p.attnNorms, buf)

		var set *Set
		if i == 0 {
			set, err = p.pipeNorm.NewSet([]*Buffer{p.xs, p.none, buf, p.none, p.none, p.normed, p.normedQ, p.normedS})
		} else {
			set, err = p.pipeNorm.NewSet([]*Buffer{p.resid, p.ffnOut, buf, p.none, p.xs, p.normed, p.normedQ, p.normedS})
		}
		if err != nil {
			return err
		}
		p.setAttnNorms = append(p.setAttnNorms, set)

		probe, err := p.pipeNorm.NewSet([]*Buffer{p.xs, p.none, buf, p.none, p.none, p.normed, p.normedQ, p.normedS})
		if err != nil {
			return err
		}
		p.setProbeNorms = append(p.setProbeNorms, probe)
	}

	for i, f := range ffnNorms {
		buf, err := p.upload(asBytes(f))
		if err != nil {
			return err
		}
		p.ffnNorms = append(p.ffnNorms, buf)

		var subOut *Buffer
		if isSSM[i] {
			subOut = p.ssmBlocks[i].outBuf
		} else {
			subOut = p.attnBlocks[i].outBuf
		}

		// resid = xs + subOut, and the feed forward's input is its norm.
		set, err := p.pipeNorm.NewSet([]*Buffer{p.xs, subOut, buf, p.none, p.resid, p.ffnNorm, p.ffnNormQ, p.ffnNormS})
		if err != nil {
			return err
		}
		p.setFFNNorms = append(p.setFFNNorms, set)
	}

	if p.outNorm, err = p.upload(asBytes(outNorm)); err != nil {
		return err
	}
	if p.setResidOnly, err = p.pipeNorm.NewSet([]*Buffer{p.resid, p.ffnOut, p.none, p.none, p.xs, p.none, p.none, p.none}); err != nil {
		return err
	}
	// The last block's residual is closed here for the same reason.
	if p.setFinalNorm, err = p.pipeNorm.NewSet([]*Buffer{p.resid, p.ffnOut, p.outNorm, p.none, p.xs, p.stage, p.none, p.none}); err != nil {
		return err
	}
	return nil
}

// Forward runs one token through every block and returns the hidden state
// under the model's output norm. The slice is the readback buffer itself and
// is overwritten by the next call.
func (p *QwenPipeline) Forward(x []float32, pos int) ([]float32, error) {
	out, err := p.ForwardColumns([][]float32{x}, []int{pos})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// ForwardColumns runs up to qwenWide tokens of one sequence through the whole
// stack for one reading of the weights, and returns each one's hidden state
// under the output norm.
//
// The positions must be consecutive and rising: the delta net's state and the
// attention's cache are both walked in the order given, and a column reads the
// keys the columns before it wrote.
func (p *QwenPipeline) ForwardColumns(xs [][]float32, positions []int) ([][]float32, error) {
	return p.forward(xs, positions, false)
}

// ForwardSpeculative is ForwardColumns for a pass whose last column is a draft:
// it copies the delta nets' state aside as it crosses from the committed
// columns into the drafted one, so that RestoreState can put it back if the
// draft is refused.
func (p *QwenPipeline) ForwardSpeculative(xs [][]float32, positions []int) ([][]float32, error) {
	return p.forward(xs, positions, true)
}

func (p *QwenPipeline) forward(xs [][]float32, positions []int, speculative bool) ([][]float32, error) {
	s := p.shape
	columns := len(xs)
	if columns == 0 || columns > qwenWide {
		return nil, fmt.Errorf("vk: a qwen pass carries one to %d columns, given %d", qwenWide, columns)
	}
	if len(positions) != columns {
		return nil, fmt.Errorf("vk: %d columns need %d positions, given %d", columns, columns, len(positions))
	}
	stream := p.xin.Floats()
	pos := unsafe.Slice((*uint32)(unsafe.Pointer(&p.posIn.Bytes()[0])), qwenWide)
	for c, x := range xs {
		if positions[c] >= s.MaxContext {
			return nil, fmt.Errorf("vk: position %d is past the %d the pipeline was built for", positions[c], s.MaxContext)
		}
		if c > 0 && positions[c] != positions[c-1]+1 {
			return nil, fmt.Errorf("vk: a pass needs consecutive positions, given %v", positions)
		}
		copy(stream[c*s.Dim:(c+1)*s.Dim], x)
		pos[c] = uint32(positions[c])
	}

	snapAt := int(noSnapshot)
	key := columns
	if speculative {
		if columns < 2 {
			return nil, fmt.Errorf("vk: a speculative pass needs a committed column and a drafted one")
		}
		snapAt = columns - 1
		key = -columns
	}
	if p.pass == nil {
		p.pass = map[int]*Program{}
	}
	prog, ok := p.pass[key]
	if !ok {
		var err error
		if prog, err = p.d.Compile(func(r *Recorder) { p.record(r, columns, snapAt) }); err != nil {
			return nil, err
		}
		p.pass[key] = prog
	}
	if err := prog.Run(); err != nil {
		return nil, err
	}

	out := make([][]float32, columns)
	normed := p.stage.Floats()
	for c := range out {
		out[c] = normed[c*s.Dim : (c+1)*s.Dim]
	}
	return out, nil
}

// record lays down one token's whole pass. It is separate from Forward so that
// the same sequence can be compiled once and replayed, which is what tells a
// recording's cost apart from the card's.
// noSnapshot is the snapAt that never matches a column.
const noSnapshot = ^uint32(0)

func (p *QwenPipeline) record(r *Recorder, columns, snapAt int) {
	s := p.shape
	dim := uint32(s.Dim)
	cols := uint32(columns)
	normFirst := normPush{n: dim, flags: normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normAttn := normPush{n: dim, flags: normAdd | normSum | normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normFFN := normAttn
	normFinal := normPush{n: dim, flags: normAdd | normSum | normGain | normFloat, eps: s.Eps, scalar: 1}

	blocks := min(len(p.isSSM), len(p.ffnBlocks))

	r.Copy(p.xs, 0, p.xin, s.Dim*columns*4)
	r.Copy(p.posBuf, 0, p.posIn, 4*columns)
	r.Barrier()

	for i := 0; i < blocks; i++ {
		push := &normAttn
		if i == 0 {
			push = &normFirst
		}
		r.DispatchColumns(p.setAttnNorms[i], 1, cols, unsafe.Pointer(push))
		r.Barrier()

		if p.isSSM[i] {
			p.recordSSM(r, p.ssmBlocks[i], columns, snapAt)
		} else {
			p.recordAttn(r, p.attnBlocks[i], columns)
		}
		r.Barrier()

		r.DispatchColumns(p.setFFNNorms[i], 1, cols, unsafe.Pointer(&normFFN))
		r.Barrier()

		p.recordFFN(r, p.ffnBlocks[i], columns)
		r.Barrier()
	}

	r.DispatchColumns(p.setFinalNorm, 1, cols, unsafe.Pointer(&normFinal))
	r.Barrier()
	r.Copy(p.hidden, 0, p.xs, s.Dim*columns*4)
}

// product dispatches a mat-vec at the width of the pass. One column runs the
// plain binary and more than one the binary built for that many, which is the
// whole of what a wide pass is: the same weights read once for every column.
func (p *QwenPipeline) product(r *Recorder, set *Set, rows, columns int, push unsafe.Pointer) {
	groups := matvecGroups(rows)
	if columns == 1 {
		r.Dispatch(set, groups, push)
		return
	}
	r.DispatchWide(set, columns, groups, push)
}

// CompileAt records a pass and keeps it. It exists for the benchmark that
// separates the recording's cost from the card's; Forward compiles its own.
func (p *QwenPipeline) CompileAt(columns int) (*Program, error) {
	if columns < 1 || columns > qwenWide {
		return nil, fmt.Errorf("vk: a qwen pass carries one to %d columns, given %d", qwenWide, columns)
	}
	return p.d.Compile(func(r *Recorder) { p.record(r, columns, int(noSnapshot)) })
}

// Probe runs the first n blocks of a token and returns the stream as it stands
// after them, un-normed. It is how a divergence is bisected down to the block
// it begins in: the CPU path can be stopped at the same place.
func (p *QwenPipeline) Probe(x []float32, pos, blocks int) ([]float32, error) {
	s := p.shape
	copy(p.xin.Floats(), x)
	p.setPos(pos)

	dim := uint32(s.Dim)
	normFirst := normPush{n: dim, flags: normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normAttn := normPush{n: dim, flags: normAdd | normSum | normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normFFN := normAttn
	normResid := normPush{n: dim, flags: normAdd | normSum, eps: s.Eps, scalar: 1}

	err := p.d.Submit(func(r *Recorder) {
		r.Copy(p.xs, 0, p.xin, s.Dim*4)
		r.Copy(p.posBuf, 0, p.posIn, 4)
		r.Barrier()
		for i := 0; i < blocks; i++ {
			push := &normAttn
			if i == 0 {
				push = &normFirst
			}
			r.Dispatch(p.setAttnNorms[i], 1, unsafe.Pointer(push))
			r.Barrier()
			if p.isSSM[i] {
				p.recordSSM(r, p.ssmBlocks[i], 1, int(noSnapshot))
			} else {
				p.recordAttn(r, p.attnBlocks[i], 1)
			}
			r.Barrier()
			r.Dispatch(p.setFFNNorms[i], 1, unsafe.Pointer(&normFFN))
			r.Barrier()
			p.recordFFN(r, p.ffnBlocks[i], 1)
			r.Barrier()
		}
		// The last block's residual is not closed by a following norm here, so
		// the probe closes it itself.
		r.Dispatch(p.setResidOnly, 1, unsafe.Pointer(&normResid))
		r.Barrier()
		r.Copy(p.hidden, 0, p.xs, s.Dim*4)
	})
	if err != nil {
		return nil, err
	}
	return p.hidden.Floats()[:s.Dim], nil
}

// ProbeMixer runs the first n blocks and then only block n's mixer, returning
// what the mixer produced before the residual.
func (p *QwenPipeline) ProbeMixer(x []float32, pos, block int) ([]float32, error) {
	s := p.shape
	copy(p.xin.Floats(), x)
	p.setPos(pos)

	dim := uint32(s.Dim)
	normFirst := normPush{n: dim, flags: normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normAttn := normPush{n: dim, flags: normAdd | normSum | normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normFFN := normAttn

	err := p.d.Submit(func(r *Recorder) {
		r.Copy(p.xs, 0, p.xin, s.Dim*4)
		r.Copy(p.posBuf, 0, p.posIn, 4)
		r.Barrier()
		for i := 0; i <= block; i++ {
			push := &normAttn
			if i == 0 {
				push = &normFirst
			}
			r.Dispatch(p.setAttnNorms[i], 1, unsafe.Pointer(push))
			r.Barrier()
			if p.isSSM[i] {
				p.recordSSM(r, p.ssmBlocks[i], 1, int(noSnapshot))
			} else {
				p.recordAttn(r, p.attnBlocks[i], 1)
			}
			r.Barrier()
			if i == block {
				break
			}
			r.Dispatch(p.setFFNNorms[i], 1, unsafe.Pointer(&normFFN))
			r.Barrier()
			p.recordFFN(r, p.ffnBlocks[i], 1)
			r.Barrier()
		}
		out := p.attnBlocks[block]
		if p.isSSM[block] {
			r.Copy(p.hidden, 0, p.ssmBlocks[block].outBuf, s.Dim*4)
		} else {
			r.Copy(p.hidden, 0, out.outBuf, s.Dim*4)
		}
	})
	if err != nil {
		return nil, err
	}
	return p.hidden.Floats()[:s.Dim], nil
}

// ProbeNormed returns the attention norm of the stream after n blocks, which
// is the mixer's input.
func (p *QwenPipeline) ProbeNormed(x []float32, pos, block int) ([]float32, error) {
	s := p.shape
	copy(p.xin.Floats(), x)
	p.setPos(pos)
	dim := uint32(s.Dim)
	normAttn := normPush{n: dim, flags: normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	err := p.d.Submit(func(r *Recorder) {
		r.Copy(p.xs, 0, p.xin, s.Dim*4)
		r.Copy(p.posBuf, 0, p.posIn, 4)
		r.Barrier()
		r.Dispatch(p.setProbeNorms[block], 1, unsafe.Pointer(&normAttn))
		r.Barrier()
		r.Copy(p.hidden, 0, p.normed, s.Dim*4)
	})
	if err != nil {
		return nil, err
	}
	return p.hidden.Floats()[:s.Dim], nil
}

// ProbeRaw runs one block's mixer over a stream the caller supplies and copies
// back one of the buffers inside it, by name. It exists for the tests that
// bisect a divergence below the block.
func (p *QwenPipeline) ProbeRaw(x []float32, pos, block int, what string) ([]float32, int, error) {
	s := p.shape
	copy(p.xin.Floats(), x)
	dim := uint32(s.Dim)
	normAttn := normPush{n: dim, flags: normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}

	p.setPos(pos)
	var pick *Buffer
	n := s.Dim
	switch what {
	case "normq":
		pick, n = p.normedQ, s.Dim/4
	case "norms":
		pick, n = p.normedS, s.Dim/16
	case "normed":
		pick, n = p.normed, s.Dim
	}
	if pick != nil {
		// nothing further
	} else if p.isSSM[block] {
		b := p.ssmBlocks[block]
		switch what {
		case "qkv":
			pick, n = b.qkvBuf, s.ConvDim
		case "gate":
			pick, n = b.gateZBuf, s.Inner
		case "alpha":
			pick, n = b.alphaBuf, s.Rank
		case "beta":
			pick, n = b.betaBuf, s.Rank
		case "conv":
			pick, n = b.convOut, s.ConvDim
		case "y":
			pick, n = b.ySSM, s.Inner
		case "out":
			pick, n = b.outBuf, s.Dim
		}
	} else {
		b := p.attnBlocks[block]
		switch what {
		case "q":
			pick, n = b.qIn, s.qFullDim()
		case "k":
			pick, n = b.kIn, s.kvDim()
		case "v":
			pick, n = b.vIn, s.kvDim()
		case "qrope":
			pick, n = b.qOut, s.qDim()
		case "mix":
			pick, n = b.attnOut, s.qDim()
		case "out":
			pick, n = b.outBuf, s.Dim
		}
	}
	if pick == nil {
		return nil, 0, fmt.Errorf("vk: no buffer named %q in block %d", what, block)
	}
	if n > s.Dim {
		if p.probe == nil {
			var err error
			if p.probe, err = p.d.Readback(s.FFN*4, bufferUsageStorage); err != nil {
				return nil, 0, err
			}
		}
	}
	dst := p.hidden
	if n > s.Dim {
		dst = p.probe
	}

	err := p.d.Submit(func(r *Recorder) {
		r.Copy(p.xs, 0, p.xin, s.Dim*4)
		r.Copy(p.posBuf, 0, p.posIn, 4)
		r.Barrier()
		r.Dispatch(p.setProbeNorms[block], 1, unsafe.Pointer(&normAttn))
		r.Barrier()
		if p.isSSM[block] {
			p.recordSSM(r, p.ssmBlocks[block], 1, int(noSnapshot))
		} else {
			p.recordAttn(r, p.attnBlocks[block], 1)
		}
		r.Barrier()
		r.Copy(dst, 0, pick, n*4)
	})
	if err != nil {
		return nil, 0, err
	}
	return dst.Floats()[:n], n, nil
}

func (p *QwenPipeline) setPos(pos int) {
	w := unsafe.Slice((*uint32)(unsafe.Pointer(&p.posIn.Bytes()[0])), qwenWide)
	w[0] = uint32(pos)
}

// Hidden is the last pass's state before the output norm. The
// multi-token-prediction block reads it, and so does the block after it.
func (p *QwenPipeline) Hidden() []float32 { return p.HiddenColumn(0) }

// HiddenColumn is the same for one column of a wider pass. The readback holds
// every column of the pass that wrote it, and the last of them is the one a
// caller carrying a conversation forward wants.
func (p *QwenPipeline) HiddenColumn(c int) []float32 {
	dim := p.shape.Dim
	return p.hidden.Floats()[c*dim : (c+1)*dim]
}

func (p *QwenPipeline) recordSSM(r *Recorder, b *qwenSSMBlock, columns, snapAt int) {
	s := p.shape
	qkv := moePush{dim: uint32(s.ConvDim), ffn: uint32(s.Dim), used: 1}
	gate := moePush{dim: uint32(s.Inner), ffn: uint32(s.Dim), used: 1}
	small := matvecKPush{Dim: uint32(s.Rank), FFN: uint32(s.Dim)}
	out := matvecKPush{Dim: uint32(s.Dim), FFN: uint32(s.Inner)}
	conv := ssmConvPush{Channels: uint32(s.ConvDim), Kernel: 4, Columns: uint32(columns), SnapAt: uint32(snapAt)}
	scan := ssmScanPush{
		NumHeads:  uint32(s.Rank),
		StateSize: uint32(s.StateSize),
		Eps:       s.Eps,
		Scale:     float32(1 / sqrtOf(s.StateSize)),
		Columns:   uint32(columns),
		ConvDim:   uint32(s.ConvDim),
		Inner:     uint32(s.Inner),
		Rank:      uint32(s.Rank),
		SnapAt:    uint32(snapAt),
	}

	p.product(r, b.setQKV, s.ConvDim, columns, unsafe.Pointer(&qkv))
	p.product(r, b.setGate, s.Inner, columns, unsafe.Pointer(&gate))
	p.product(r, b.setAlpha, s.Rank, columns, unsafe.Pointer(&small))
	p.product(r, b.setBeta, s.Rank, columns, unsafe.Pointer(&small))
	r.Barrier()

	// The convolution and the scan carry the columns inside themselves: both
	// hold state that runs from one token to the next, so a column cannot
	// start before the one before it has finished.
	r.Dispatch(b.setConv, uint32((s.ConvDim+255)/256), unsafe.Pointer(&conv))
	r.Barrier()

	r.Dispatch(b.setScan, uint32(s.Rank), unsafe.Pointer(&scan))
	r.Barrier()

	p.product(r, b.setOut, s.Dim, columns, unsafe.Pointer(&out))
}

func (p *QwenPipeline) recordAttn(r *Recorder, b *qwenAttnBlock, columns int) {
	s := p.shape
	q := moePush{dim: uint32(s.qFullDim()), ffn: uint32(s.Dim), used: 1}
	kv := moePush{dim: uint32(s.kvDim()), ffn: uint32(s.Dim), used: 1}
	out := matvecKPush{Dim: uint32(s.Dim), FFN: uint32(s.qDim())}
	prep := attnPrepPush{
		MaxContext: uint32(s.MaxContext),
		Heads:      uint32(s.Heads),
		KVHeads:    uint32(s.KVHeads),
		RoPEBase:   s.RoPEBase,
		Eps:        s.Eps,
		RoPEDims:   uint32(s.RoPEDims),
	}
	gqa := attnGQAPush{
		MaxContext: uint32(s.MaxContext),
		HeadsPerKV: uint32(s.Heads / s.KVHeads),
		Heads:      uint32(s.Heads),
		Scale:      float32(1 / sqrtOf(s.HeadDim)),
	}

	p.product(r, b.setQ, s.qFullDim(), columns, unsafe.Pointer(&q))
	p.product(r, b.setK, s.kvDim(), columns, unsafe.Pointer(&kv))
	p.product(r, b.setV, s.kvDim(), columns, unsafe.Pointer(&kv))
	r.Barrier()

	// Every column's keys and values go into the cache before any column reads
	// them, which is why this is a dispatch of its own and not the head of the
	// one below: a barrier inside a workgroup does not order two workgroups.
	r.DispatchColumns(b.setPrep, uint32(s.Heads+s.KVHeads), uint32(columns), unsafe.Pointer(&prep))
	r.Barrier()

	r.DispatchColumns(b.setGQA, uint32(s.Heads), uint32(columns), unsafe.Pointer(&gqa))
	r.Barrier()

	p.product(r, b.setO, s.Dim, columns, unsafe.Pointer(&out))
}

func (p *QwenPipeline) recordFFN(r *Recorder, b *qwenFFNBlock, columns int) {
	s := p.shape
	up := moePush{dim: uint32(s.FFN), ffn: uint32(s.Dim), used: 1}
	// The activation is elementwise and the columns lie end to end, so a wide
	// pass is the same kernel over more of the buffer.
	act := swigluPush{N: uint32(s.FFN * columns)}
	down := matvecKPush{Dim: uint32(s.Dim), FFN: uint32(s.FFN)}

	p.product(r, b.setGate, s.FFN, columns, unsafe.Pointer(&up))
	p.product(r, b.setUp, s.FFN, columns, unsafe.Pointer(&up))
	r.Barrier()

	r.Dispatch(p.setAct, uint32((s.FFN*columns+255)/256), unsafe.Pointer(&act))
	r.Barrier()

	p.product(r, b.setDown, s.Dim, columns, unsafe.Pointer(&down))
}

// ResetState clears what a conversation accumulates: the delta nets' state
// matrices and their convolution windows. The attention caches need no
// clearing, because a pass only ever reads the positions it has written.
func (p *QwenPipeline) ResetState() error {
	return p.d.Submit(func(r *Recorder) {
		for _, b := range p.ssmBlocks {
			r.Fill(b.ssmState, 0)
			r.Fill(b.convState, 0)
		}
	})
}

func (p *QwenPipeline) Close() {
	for _, prog := range p.pass {
		prog.Close()
	}
	p.pass = nil
	if p.restoreProg != nil {
		p.restoreProg.Close()
		p.restoreProg = nil
	}
	if p.mtp != nil {
		if p.mtp.pass != nil {
			p.mtp.pass.Close()
		}
		p.mtp.ehIn.Close()
		p.mtp = nil
	}
	for _, b := range p.owned {
		b.Close()
	}
	p.owned = nil
	if p.xin != nil {
		p.xin.Close()
	}
	if p.stage != nil {
		p.stage.Close()
	}
	if p.hidden != nil {
		p.hidden.Close()
	}
	if p.probe != nil {
		p.probe.Close()
	}
	if p.posIn != nil {
		p.posIn.Close()
	}
	for _, pl := range []*Pipeline{
		p.pipeNorm, p.pipeMatvec, p.pipeMatQ40, p.pipeMatQ41, p.pipeMatQ5K,
		p.pipeMatF32, p.pipeSwiglu, p.pipeConv, p.pipeScan, p.pipeAttnPrep, p.pipeAttnGQA,
		p.pipeMatQ80,
	} {
		if pl != nil {
			pl.Close()
		}
	}
}

// The multi-token-prediction block, which is a Qwen3.5 block with a two-input
// front: the embedding of the token just decided and the trunk's hidden state
// for the one before it, each normed, concatenated, and projected back down to
// one hidden state. It shares the stream buffers with the trunk because the two
// never run at once — a draft is made, and only then is it verified.
//
// It is on the card for the same reason the trunk is. On the processor the
// block is 14 ms, which is half a token, and a draft that costs half a token
// cannot save one.

type QwenMTPData struct {
	// EHProj is [Dim, 2*Dim] and Q8_0 in the checkpoints seen so far. Its bytes
	// go up as the file holds them: shaders/matvec_q80.comp reads the format's
	// own interleaving of a scale and thirty-two magnitudes.
	EHProj []byte
	Attn   QwenAttnData
	FFN    QwenFFNData

	AttnNorm []float32
	FFNNorm  []float32
	HeadNorm []float32
}

type qwenMTPBlock struct {
	wEH   *Buffer
	ehIn  *Buffer
	ehBuf *Buffer

	attn *qwenAttnBlock
	ffn  *qwenFFNBlock

	attnNorm *Buffer
	ffnNorm  *Buffer
	headNorm *Buffer

	setEH       *Set
	setAttnNorm *Set
	setFFNNorm  *Set
	setFinal    *Set

	pass *Program
}

// AddMTPBlock uploads the prediction block. It must come after SetNorms, which
// is what the trunk's own stream bindings wait for.
func (p *QwenPipeline) AddMTPBlock(d QwenMTPData) error {
	s := p.shape
	b := &qwenMTPBlock{}
	var err error

	if p.pipeMatQ80 == nil {
		if p.pipeMatQ80, err = p.d.NewPipeline(matvecQ80SPIRV, 3, uint32(unsafe.Sizeof(matvecKPush{}))); err != nil {
			return err
		}
	}
	if b.wEH, err = p.upload(d.EHProj); err != nil {
		return err
	}
	if b.ehIn, err = p.d.Host(s.Dim*2*4, bufferUsageStorage|bufferUsageTransferSrc); err != nil {
		return err
	}
	if b.ehBuf, err = p.local(s.Dim * 2 * 4); err != nil {
		return err
	}
	if b.attn, err = p.newAttnBlock(d.Attn); err != nil {
		return err
	}
	if b.ffn, err = p.newFFNBlock(d.FFN); err != nil {
		return err
	}
	for _, u := range []struct {
		into **Buffer
		data []float32
	}{{&b.attnNorm, d.AttnNorm}, {&b.ffnNorm, d.FFNNorm}, {&b.headNorm, d.HeadNorm}} {
		if *u.into, err = p.upload(asBytes(u.data)); err != nil {
			return err
		}
	}

	if b.setEH, err = p.pipeMatQ80.NewSet([]*Buffer{b.wEH, b.ehBuf, p.xs}); err != nil {
		return err
	}
	if b.setAttnNorm, err = p.pipeNorm.NewSet([]*Buffer{p.xs, p.none, b.attnNorm, p.none, p.none, p.normed, p.normedQ, p.normedS}); err != nil {
		return err
	}
	if b.setFFNNorm, err = p.pipeNorm.NewSet([]*Buffer{p.xs, b.attn.outBuf, b.ffnNorm, p.none, p.resid, p.ffnNorm, p.ffnNormQ, p.ffnNormS}); err != nil {
		return err
	}
	if b.setFinal, err = p.pipeNorm.NewSet([]*Buffer{p.resid, p.ffnOut, b.headNorm, p.none, p.xs, p.stage, p.none, p.none}); err != nil {
		return err
	}

	p.mtp = b
	return nil
}

func (p *QwenPipeline) HasMTP() bool { return p.mtp != nil }

// Columns is how many tokens one pass can carry.
func (p *QwenPipeline) Columns() int { return qwenWide }

// WidthFor is the widest pass that fits n remaining tokens. A run is read in
// passes of these rather than one width and a ragged tail of single columns:
// the mat-vec binaries exist at four widths and the largest that fits wins.
func (p *QwenPipeline) WidthFor(n int) int {
	for _, w := range qwenWidths {
		if w <= n {
			return w
		}
	}
	return 1
}

// DraftMTP runs the prediction block over one already-normed and concatenated
// [enorm(embed) | hnorm(hidden)] pair and returns the hidden state under the
// block's own head norm, ready for the logit head.
//
// The two norms are left to the caller because they are two passes over five
// thousand floats — nothing beside the two hundred and sixty megabytes of
// weights this reads — and doing them here would mean a second buffer across
// the bus for the hidden state the caller already has.
func (p *QwenPipeline) DraftMTP(eh []float32, pos int) ([]float32, error) {
	s := p.shape
	if p.mtp == nil {
		return nil, fmt.Errorf("vk: the pipeline carries no prediction block")
	}
	if len(eh) != s.Dim*2 {
		return nil, fmt.Errorf("vk: the prediction block takes %d floats, given %d", s.Dim*2, len(eh))
	}
	if pos >= s.MaxContext {
		return nil, fmt.Errorf("vk: position %d is past the %d the pipeline was built for", pos, s.MaxContext)
	}
	copy(p.mtp.ehIn.Floats(), eh)
	p.setPos(pos)

	if p.mtp.pass == nil {
		prog, err := p.d.Compile(p.recordMTP)
		if err != nil {
			return nil, err
		}
		p.mtp.pass = prog
	}
	if err := p.mtp.pass.Run(); err != nil {
		return nil, err
	}
	return p.stage.Floats()[:s.Dim], nil
}

func (p *QwenPipeline) recordMTP(r *Recorder) {
	s := p.shape
	b := p.mtp
	dim := uint32(s.Dim)

	eh := matvecKPush{Dim: dim, FFN: uint32(s.Dim * 2)}
	normAttn := normPush{n: dim, flags: normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normFFN := normPush{n: dim, flags: normAdd | normSum | normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normFinal := normPush{n: dim, flags: normAdd | normSum | normGain | normFloat, eps: s.Eps, scalar: 1}

	r.Copy(b.ehBuf, 0, b.ehIn, s.Dim*2*4)
	r.Copy(p.posBuf, 0, p.posIn, 4)
	r.Barrier()

	r.Dispatch(b.setEH, matvecGroups(s.Dim), unsafe.Pointer(&eh))
	r.Barrier()

	r.Dispatch(b.setAttnNorm, 1, unsafe.Pointer(&normAttn))
	r.Barrier()

	p.recordAttn(r, b.attn, 1)
	r.Barrier()

	r.Dispatch(b.setFFNNorm, 1, unsafe.Pointer(&normFFN))
	r.Barrier()

	p.recordFFN(r, b.ffn, 1)
	r.Barrier()

	r.Dispatch(b.setFinal, 1, unsafe.Pointer(&normFinal))
	r.Barrier()
}

// ResetMTPCache clears the prediction block's key-value cache, which a new
// conversation needs and a rejected draft does not: the next draft at that
// position writes over it.
func (p *QwenPipeline) ResetMTPCache() error {
	if p.mtp == nil {
		return nil
	}
	return p.d.Submit(func(r *Recorder) {
		r.Fill(p.mtp.attn.kCache, 0)
		r.Fill(p.mtp.attn.vCache, 0)
	})
}

// Speculation on a recurrent model has a difficulty an attention-only one does
// not: a key written at a position the draft turned out not to occupy is simply
// overwritten by the token that does occupy it, but a delta net's state matrix
// has already absorbed the wrong token and there is nothing to overwrite it
// with. So the state is copied aside before a pass that carries a draft, and
// copied back when the draft is refused.
//
// It is a hundred and fifty-one megabytes a snapshot — forty-eight blocks of
// forty-eight 128x128 matrices — which is half a millisecond of a thirty-two
// millisecond pass. The convolution windows go with it, five more.

// RestoreState puts the delta nets back to the state a speculative pass copied
// aside — the one after the committed columns and before the drafted one. It is
// what a refused draft needs, and only that: the attention's keys at the
// refused position are overwritten by the token that does occupy it.
func (p *QwenPipeline) RestoreState() error {
	if p.restoreProg == nil {
		prog, err := p.d.Compile(p.recordRestore)
		if err != nil {
			return err
		}
		p.restoreProg = prog
	}
	return p.restoreProg.Run()
}

func (p *QwenPipeline) recordRestore(r *Recorder) {
	for _, b := range p.ssmBlocks {
		r.Copy(b.ssmState, 0, b.shadowState, b.shadowState.Size())
		r.Copy(b.convState, 0, b.shadowConv, b.shadowConv.Size())
	}
	r.Barrier()
}
