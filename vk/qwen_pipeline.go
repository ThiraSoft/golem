package vk

// Every block of Qwen3.8 on the card, one submission to a token.
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
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q5k.comp -o shaders/matvec_q5k.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/swiglu_act.comp -o shaders/swiglu_act.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/quant_q80.comp -o shaders/quant_q80.spv
//go:generate glslc -O -DBM=64 -DBN=64 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_q5k.comp -o shaders/matmul_q5k.spv
//go:generate glslc -O -DQ41 -DBM=64 -DBN=64 --target-env=vulkan1.1 -fshader-stage=compute shaders/matmul_q5k.comp -o shaders/matmul_q41t.spv
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

// The tiled product again, for the Q4_1 weights a Q4_0 checkpoint still keeps:
// llama.cpp's quantizer leaves ffn_down at Q4_1, and it is the largest matrix
// in a block. shaders/matmul.comp's -DQ41 says what differs — twenty bytes a
// block instead of eighteen, and a minimum where the nibble's offset of eight
// was.
//

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

// narrowChunk is the widest a mat-vec is dispatched at, and it binds productK
// alone: the pipelines product serves carry the tiled binaries as well, and
// there the widest available is the one to take.
//
// Sixteen and not more, and the reason changed once the fault below it was
// found. The thirty-two-wide form used to look a third faster and answer
// wrongly, which was dispatchAt taking the tiled workgroup count for any pass
// of thirty-two or more — a mat-vec dispatched over a sixty-fourth of the rows
// it needed. With that fixed the binary is right and simply slower: 387
// positions a second at two hundred and fifty-six columns against 441 at
// sixteen, because a thread carrying thirty-two accumulators spills.
// TestVulkanWidePassMatchesTokenPath holds a wide pass to what the same tokens
// give one at a time, bit for bit, and only widths up to sixteen pass it. What
// this caps is the two projections whose weights have no tiled form here;
// everything else goes through the tiled product at these widths anyway.
const narrowChunk = 16

// quantBlock is nn.QuantBlock, said here so this file keeps its import list.
const quantBlock = 32

// qwenWide is the widest pass any of these binaries answers. Everything
// per-column is allocated for it.
const qwenWide = 512

// qwenWidths are the widths a pass may take, largest first. A run of tokens is
// cut into passes of these: a hundred positions is twelve of eight and one of
// four, and the remainder never falls back to one column at a time unless it
// is one column.
var qwenWidths = [...]int{512, 256, 128, 64, 32, 16, 8, 4, 2, 1}

//go:embed shaders/quant_q80.spv
var quantQ80SPIRV []byte

// The tiled Q5_K product. One binary whatever the width: the columns are the
// dispatch's business, not the kernel's, so unlike every other product here it
// takes no COLUMNS.
//
//go:embed shaders/matmul_q5k.spv
var matmulQ5KSPIRV []byte

// The delta net's output projection on the matrix cores.
//
// This is shaders/matmul_coop.comp under -DQ5K, and it exists because that
// projection was the one matrix in the model still read by a hand-rolled
// tile: 68ms of a 512-column pass against llama.cpp's 28, where the same
// kernel's Q4_0 form runs the feed forward's gate and up *faster* than theirs.
// A quarter of the throughput of our own best product, on the same card, for
// no reason but the format's packing — which vk/mixture.go's splitQ5_K now
// undoes on the way to the card.
//
//go:embed shaders/matmul_coop_q5k64.spv
var matmulCoopQ5K64SPIRV []byte

//go:embed shaders/matmul_coop_q5k128.spv
var matmulCoopQ5K128SPIRV []byte

//go:embed shaders/matmul_coop_q5k256.spv
var matmulCoopQ5K256SPIRV []byte

//go:embed shaders/matmul_coop_q5k512.spv
var matmulCoopQ5K512SPIRV []byte

// The same tile over Q4_1 weights, which is what this checkpoint keeps eight
// of its sixty-five down projections in.
//
//go:embed shaders/matmul_q41t.spv
var matmulQ41TSPIRV []byte

// q5kRows and q5kCols are that kernel's tile, and the three have to agree: the
// shader's BM and BN, and the workgroup count recordSSM dispatches.
const q5kRows = 64
const q5kCols = 64

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
	N       uint32
	Columns uint32
}

type attnPrepPush struct {
	MaxContext uint32
	Heads      uint32
	KVHeads    uint32
	RoPEBase   float32
	Eps        float32
	RoPEDims   uint32
	// The M-RoPE section widths, in the order the shader reads them. A field
	// out of place here is not a compile error on either side: the kernel
	// reads whatever integer lands at the offset it expects.
	Sect0 uint32
	Sect1 uint32
	Sect2 uint32
	Sect3 uint32
}

type attnGQAPush struct {
	MaxContext uint32
	HeadsPerKV uint32
	Heads      uint32
	Scale      float32
	Columns    uint32
}

// qAttnTile is shaders/qwen_attn_gqa.comp's QTILE: how many columns of the
// pass one attention workgroup answers. The keys and the values of a tile are
// read once for all of them, so this is the factor by which the attention's
// traffic falls — it is quadratic in the context either way, but the constant
// is this. The two have to agree: the grid is sized from here.
const qAttnTile = 8

// QwenShape is the geometry every block of one model shares. It is passed once
// rather than rediscovered per block because nothing in Qwen3.8 varies from
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
	// RoPESections are the four M-RoPE section widths the file declares, all
	// zero for a checkpoint that declares none. They decide which of a
	// position's axes each pair of a head turns by.
	RoPESections [4]int

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

// qkHeads is how many query-key heads the delta net's convolution carries.
// Its output is those heads' q, then their k, then one v a head of the state,
// so what is left of ConvDim once the values are taken out is two of them.
func (s QwenShape) qkHeads() int { return (s.ConvDim - s.Inner) / 2 / s.StateSize }

// scanColumns is shaders/ssm_scan.comp's COLS: how many columns of a head's
// state one workgroup of the recurrence owns. The two have to agree — the grid
// is sized from here.
const scanColumns = 8

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

	convWeight, convState   *Buffer
	shadowState, shadowConv *Buffer
	dtBias, ssmA, ssmNorm   *Buffer
	ssmState                *Buffer

	setQKV, setGate, setAlpha, setBeta *Set
	setConv, setScan, setOut           *Set
	// setNormGate is the gated RMS norm of the recurrence's output, which used
	// to be the tail of the scan and is now a pass of its own.
	setNormGate *Set
	// setOutWide is that projection through the tiled Q5_K product against the
	// output's Q8_0 form. Nil where the weights are not Q5_K.
	setOutWide *Set
}

type qwenAttnBlock struct {
	wQ, wK, wV, wO *Buffer
	qNorm, kNorm   *Buffer
	kCache, vCache *Buffer

	setQ, setK, setV *Set
	setPrep, setGQA  *Set
	setO, setOWide   *Set
}

type qwenFFNBlock struct {
	wGate, wUp, wDown       *Buffer
	setGate, setUp, setDown *Set
	// setDownWide is the same projection through the tiled product against the
	// activation's Q8_0 form, which only a pass of thirty-two columns or more
	// reaches. It is nil where the weights are Q4_1, which has no tiled form
	// here — eight blocks of sixty-five in this checkpoint, the rest Q4_0.
	setDownWide *Set
	// q41 says which of the two the weights are, because the tiled forms are
	// dispatched differently: the Q4_1 one answers a fixed tile of columns
	// where the Q4_0 one is a width-compiled binary.
	q41 bool
}

type QwenPipeline struct {
	d     *Device
	shape QwenShape
	// tl is where a recording writes the card's clock, when one is installed.
	// Qwen3.8 is the only pipeline here that never had one: every number this
	// model's performance work has ever rested on came from ablation — remove
	// a kernel, time the whole pass — which measures a difference and never a
	// share, and cannot see time that belongs to no kernel at all.
	tl *Timeline
	// coop says the card has cooperative matrices, which decides which tiled
	// product the wide projections are bound to and how many workgroups it
	// wants. vk/matmul.go owns both.
	coop bool

	pipeNorm     *Pipeline
	pipeMatvec   *Pipeline // Q4_0 against the Q8_0 activation
	pipeMatQ40   *Pipeline // Q4_0 against floats
	pipeMatQ41   *Pipeline
	pipeMatQ5K   *Pipeline
	pipeMatF32   *Pipeline
	pipeSwiglu   *Pipeline
	pipeConv     *Pipeline
	pipeScan     *Pipeline
	pipeQKNorm   *Pipeline
	pipeGate     *Pipeline
	pipeAttnPrep *Pipeline
	pipeAttnGQA  *Pipeline
	pipeMatQ80   *Pipeline // the prediction block's front projection
	pipeQuant    *Pipeline // floats to their Q8_0 form
	pipeMatT5K   *Pipeline // the tiled Q5_K product, which only a wide pass reaches
	pipeMatT41   *Pipeline // the same tile over Q4_1 weights

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
	// The rotation's three axes, beside the cache index rather than folded
	// into it. shaders/qwen_attn_prep.comp says why they cannot be one number.
	mposIn  *Buffer
	mposBuf *Buffer
	pass    map[int]*Program

	// The scratch a block passes through, which is the pipeline's and not the
	// block's: sixty-four blocks go through one command buffer with a barrier
	// between them, so no two of them are ever in flight and one set of these
	// serves them all. A set a block, at a hundred and twenty-eight columns,
	// was a gigabyte — enough to push the logit head off the card, which
	// showed up as a token costing 200ms instead of 34.
	convOut, qkvBuf *Buffer
	// qkNorm is the delta net's q and k for a whole pass, L2 normed: q at the
	// head of a column and k two thousand and forty-eight floats after it.
	qkNorm             *Buffer
	gateZBuf, ySSM     *Buffer
	alphaBuf, betaBuf  *Buffer
	qIn, kIn, vIn      *Buffer
	qOut, attnOut      *Buffer
	mixOut             *Buffer // whichever mixer this block has, into the norm
	actQ, actS         *Buffer // the feed forward's activation in its Q8_0 form
	ySSMQ, ySSMS       *Buffer // the delta net's output in its Q8_0 form
	attnOutQ, attnOutS *Buffer // the attention's mix in its Q8_0 form

	resid    *Buffer
	normed   *Buffer
	normedQ  *Buffer
	normedS  *Buffer
	ffnNorm  *Buffer
	ffnNormQ *Buffer
	ffnNormS *Buffer
	none     *Buffer

	gateBuf      *Buffer
	upBuf        *Buffer
	actBuf       *Buffer
	ffnOut       *Buffer
	setAct       *Set
	setQuantY    *Set
	setQKNorm    *Set
	setQuantAttn *Set

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
		coop:       d.Coopmat(),
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
		{&p.pipeSwiglu, swigluActSPIRV, 5, unsafe.Sizeof(swigluPush{})},
		{&p.pipeQuant, quantQ80SPIRV, 3, unsafe.Sizeof(swigluPush{})},
		{&p.pipeMatT5K, matmulQ5KSPIRV, 4, unsafe.Sizeof(moePush{})},
		{&p.pipeMatT41, matmulQ41TSPIRV, 4, unsafe.Sizeof(moePush{})},
		{&p.pipeConv, ssmConv1dSPIRV, 5, unsafe.Sizeof(ssmConvPush{})},
		{&p.pipeScan, ssmScanSPIRV, 9, unsafe.Sizeof(ssmScanPush{})},
		{&p.pipeQKNorm, ssmQKNormSPIRV, 2, unsafe.Sizeof(ssmQKNormPush{})},
		{&p.pipeGate, ssmGateSPIRV, 3, unsafe.Sizeof(ssmGatePush{})},
		{&p.pipeAttnPrep, qwenAttnPrepSPIRV, 10, unsafe.Sizeof(attnPrepPush{})},
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
	// And the tiled product on the one pipeline whose weights are Q4_0 against
	// the Q8_0 activation, which is what it reads. Its bindings and its push
	// block are the mat-vec's, so no set has to be made again — vk/mixture.go
	// binds the same two kernels to one pipeline for the same reason.
	//
	// It is the largest projections that go through it: the delta net's q+k+v
	// and its gate, the attention's three, and the feed forward's gate and up.
	// A mat-vec reads a weight once for sixteen columns and stops there,
	// because the accumulator a thread carries a column in stops fitting in
	// registers; the tiled product stages both operands and keeps a tile of
	// the answer, and llama.cpp draws the same line at eight.
	tiled := []struct {
		columns int
		spirv   []byte
	}{
		{tiledColumns, matmulWide32SPIRV},
		{64, matmulWide64SPIRV},
		{128, matmulWidest128SPIRV},
		{256, matmulWidest256SPIRV},
		{wideColumns, matmulWide()},
	}
	wave := uint32(0)
	if p.coop {
		wave = coopmatWave
		tiled = []struct {
			columns int
			spirv   []byte
		}{
			{tiledColumns, matmulCoop32SPIRV},
			{64, matmulCoop64SPIRV},
			{128, matmulCoop128SPIRV},
			{256, matmulCoop256SPIRV},
			{wideColumns, matmulCoop512SPIRV},
		}
	}
	for _, w := range tiled {
		if err := p.pipeMatvec.WideWave(w.columns, w.spirv, wave); err != nil {
			return nil, err
		}
	}
	// And the same kernel over Q5_K, on the one pipeline whose weights are
	// packed by splitQ5_K. Only a card with matrix cores gets it: without them
	// the delta net's output projection stays on the mat-vec, which reads the
	// same packing.
	if p.coop {
		for _, w := range []struct {
			columns int
			spirv   []byte
		}{
			{64, matmulCoopQ5K64SPIRV},
			{128, matmulCoopQ5K128SPIRV},
			{256, matmulCoopQ5K256SPIRV},
			{wideColumns, matmulCoopQ5K512SPIRV},
		} {
			if err := p.pipeMatT5K.WideWave(w.columns, w.spirv, coopmatWave); err != nil {
				return nil, err
			}
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
	// One position a column of the widest pass. It was sixty-four bytes when a
	// pass carried two, which is sixteen of them — and a pass of thirty-two
	// then wrote past the end of it and copied twice the buffer's length into
	// the device's. That answered rather than failing: every wide pass read
	// somebody else's positions.
	if p.posIn, err = d.Host(qwenWide*4, bufferUsageStorage|bufferUsageTransferSrc); err != nil {
		return nil, err
	}
	// Four uints a column, against the position's one.
	if p.mposIn, err = d.Host(qwenWide*16, bufferUsageStorage|bufferUsageTransferSrc); err != nil {
		return nil, err
	}
	if p.mposBuf, err = d.Local(qwenWide*16, bufferUsageStorage|bufferUsageTransferDst); err != nil {
		return nil, err
	}
	if p.posBuf, err = d.Local(qwenWide*4, bufferUsageStorage); err != nil {
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
	// The same activation in eight bits, which is what the tiled product reads.
	if p.actQ, err = p.local(shape.FFN * qwenWide); err != nil {
		return nil, err
	}
	if p.actS, err = p.local(2 * (shape.FFN / quantBlock) * qwenWide * 4); err != nil {
		return nil, err
	}
	// And the delta net's output in the same two forms, for the same reason:
	// its projection is Q5_K and the tiled product wants eight bits.
	if p.ySSMQ, err = p.local(shape.Inner * qwenWide); err != nil {
		return nil, err
	}
	if p.ySSMS, err = p.local(2 * (shape.Inner / quantBlock) * qwenWide * 4); err != nil {
		return nil, err
	}
	// And the attention's mix, whose output projection is Q4_0 and wants the
	// same eight bits.
	if p.attnOutQ, err = p.local(shape.qDim() * qwenWide); err != nil {
		return nil, err
	}
	if p.attnOutS, err = p.local(2 * (shape.qDim() / quantBlock) * qwenWide * 4); err != nil {
		return nil, err
	}
	// The mixer's scratch, one set for every block. See the field comment.
	for _, l := range []struct {
		into  **Buffer
		bytes int
	}{
		{&p.convOut, shape.ConvDim * qwenWide * 4}, {&p.qkvBuf, shape.ConvDim * qwenWide * 4},
		{&p.qkNorm, 4096 * qwenWide * 4},
		{&p.gateZBuf, shape.Inner * qwenWide * 4}, {&p.ySSM, shape.Inner * qwenWide * 4},
		{&p.alphaBuf, shape.Rank * qwenWide * 4}, {&p.betaBuf, shape.Rank * qwenWide * 4},
		{&p.qIn, shape.qFullDim() * qwenWide * 4},
		{&p.kIn, shape.kvDim() * qwenWide * 4}, {&p.vIn, shape.kvDim() * qwenWide * 4},
		{&p.qOut, shape.qDim() * qwenWide * 4}, {&p.attnOut, shape.qDim() * qwenWide * 4},
		{&p.mixOut, shape.Dim * qwenWide * 4},
	} {
		if *l.into, err = p.local(l.bytes); err != nil {
			return nil, err
		}
	}
	if p.setAct, err = p.pipeSwiglu.NewSet([]*Buffer{p.gateBuf, p.upBuf, p.actBuf, p.actQ, p.actS}); err != nil {
		return nil, err
	}
	if p.setQKNorm, err = p.pipeQKNorm.NewSet([]*Buffer{p.convOut, p.qkNorm}); err != nil {
		return nil, err
	}
	if p.setQuantY, err = p.pipeQuant.NewSet([]*Buffer{p.ySSM, p.ySSMQ, p.ySSMS}); err != nil {
		return nil, err
	}
	if p.setQuantAttn, err = p.pipeQuant.NewSet([]*Buffer{p.attnOut, p.attnOutQ, p.attnOutS}); err != nil {
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
	case d.OutIsQ5K:
		// splitQ5_K, not the file's own superblocks: both readers of this
		// matrix — the mat-vec a token runs and the cooperative tile a prompt
		// runs — want a block of thirty-two that stands alone.
		b.wOut, err = p.upload(splitQ5_K(d.WOut, s.Dim, s.Inner))
	case d.OutIsF32:
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
	if b.setQKV, err = p.pipeMatvec.NewSet([]*Buffer{b.wQKV, p.normedQ, p.normedS, p.qkvBuf}); err != nil {
		return err
	}
	if b.setGate, err = p.pipeMatvec.NewSet([]*Buffer{b.wGate, p.normedQ, p.normedS, p.gateZBuf}); err != nil {
		return err
	}
	if b.setAlpha, err = p.pipeMatF32.NewSet([]*Buffer{b.wAlpha, p.normed, p.alphaBuf}); err != nil {
		return err
	}
	if b.setBeta, err = p.pipeMatF32.NewSet([]*Buffer{b.wBeta, p.normed, p.betaBuf}); err != nil {
		return err
	}
	if b.setConv, err = p.pipeConv.NewSet([]*Buffer{b.convWeight, p.qkvBuf, b.convState, p.convOut, b.shadowConv}); err != nil {
		return err
	}
	if b.setScan, err = p.pipeScan.NewSet([]*Buffer{p.convOut, p.qkNorm, p.alphaBuf, b.dtBias, b.ssmA, p.betaBuf, b.ssmState, p.ySSM, b.shadowState}); err != nil {
		return err
	}
	if b.setNormGate, err = p.pipeGate.NewSet([]*Buffer{p.ySSM, p.gateZBuf, b.ssmNorm}); err != nil {
		return err
	}
	outPipe := p.pipeMatQ41
	switch {
	case d.OutIsF32:
		outPipe = p.pipeMatF32
	case d.OutIsQ5K:
		outPipe = p.pipeMatQ5K
	}
	// The tiled form of it, which only a Q5_K block has and only a wide pass
	// reaches. It reads the delta net's output in eight bits where the mat-vec
	// reads it in floats, so a token draws exactly as it did.
	if d.OutIsQ5K && p.coop {
		if b.setOutWide, err = p.pipeMatT5K.NewSet([]*Buffer{b.wOut, p.ySSMQ, p.ySSMS, p.mixOut}); err != nil {
			return err
		}
	}
	if b.setOut, err = outPipe.NewSet([]*Buffer{b.wOut, p.ySSM, p.mixOut}); err != nil {
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
	if b.setQ, err = p.pipeMatvec.NewSet([]*Buffer{b.wQ, p.normedQ, p.normedS, p.qIn}); err != nil {
		return nil, err
	}
	if b.setK, err = p.pipeMatvec.NewSet([]*Buffer{b.wK, p.normedQ, p.normedS, p.kIn}); err != nil {
		return nil, err
	}
	if b.setV, err = p.pipeMatvec.NewSet([]*Buffer{b.wV, p.normedQ, p.normedS, p.vIn}); err != nil {
		return nil, err
	}
	if b.setPrep, err = p.pipeAttnPrep.NewSet([]*Buffer{p.qIn, p.kIn, p.vIn, b.qNorm, b.kNorm, p.qOut, b.kCache, b.vCache, p.posBuf, p.mposBuf}); err != nil {
		return nil, err
	}
	if b.setGQA, err = p.pipeAttnGQA.NewSet([]*Buffer{p.qOut, b.kCache, b.vCache, p.qIn, p.attnOut, p.posBuf}); err != nil {
		return nil, err
	}
	if b.setO, err = p.pipeMatQ40.NewSet([]*Buffer{b.wO, p.attnOut, p.mixOut}); err != nil {
		return nil, err
	}
	// And through the tiled product against that mix in eight bits, which only
	// a wide pass reaches. The output projection is Q4_0 like the three above
	// it; it read floats only because nothing had quantized the mix.
	if b.setOWide, err = p.pipeMatvec.NewSet([]*Buffer{b.wO, p.attnOutQ, p.attnOutS, p.mixOut}); err != nil {
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
	b.q41 = d.DownQ4_1
	wide := p.pipeMatvec // Q4_0 against the Q8_0 activation
	if d.DownQ4_1 {
		wide = p.pipeMatT41
	}
	if b.setDownWide, err = wide.NewSet([]*Buffer{b.wDown, p.actQ, p.actS, p.ffnOut}); err != nil {
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
			subOut = p.mixOut
		} else {
			subOut = p.mixOut
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
	return p.forward(xs, RunColumns(positions), false)
}

// QwenPlace is one column's cache index and the three axes its rotation reads.
// Pos is what the cache and the visible range count in; the triple is the
// rotation's alone, and an image is what makes the two differ.
type QwenPlace struct {
	Pos     int
	T, H, W int
}

// RunColumns is the places of a run of text, where every axis follows the
// cache index. It is what every caller but an image gives.
func RunColumns(positions []int) []QwenPlace {
	at := make([]QwenPlace, len(positions))
	for i, q := range positions {
		at[i] = QwenPlace{Pos: q, T: q, H: q, W: q}
	}
	return at
}

// ForwardPlaces is ForwardColumns for a pass whose axes do not all follow the
// cache index.
func (p *QwenPipeline) ForwardPlaces(xs [][]float32, at []QwenPlace) ([][]float32, error) {
	return p.forward(xs, at, false)
}

// ForwardSpeculative is ForwardColumns for a pass whose last column is a draft:
// it copies the delta nets' state aside as it crosses from the committed
// columns into the drafted one, so that RestoreState can put it back if the
// draft is refused.
func (p *QwenPipeline) ForwardSpeculative(xs [][]float32, positions []int) ([][]float32, error) {
	return p.forward(xs, RunColumns(positions), true)
}

// ForwardSpeculativeAt is ForwardSpeculative for places whose axes do not
// follow the cache index.
func (p *QwenPipeline) ForwardSpeculativeAt(xs [][]float32, at []QwenPlace) ([][]float32, error) {
	return p.forward(xs, at, true)
}

func (p *QwenPipeline) forward(xs [][]float32, at []QwenPlace, speculative bool) ([][]float32, error) {
	s := p.shape
	columns := len(xs)
	if columns == 0 || columns > qwenWide {
		return nil, fmt.Errorf("vk: a qwen pass carries one to %d columns, given %d", qwenWide, columns)
	}
	if len(at) != columns {
		return nil, fmt.Errorf("vk: %d columns need %d positions, given %d", columns, columns, len(at))
	}
	stream := p.xin.Floats()
	pos := unsafe.Slice((*uint32)(unsafe.Pointer(&p.posIn.Bytes()[0])), qwenWide)
	mpos := unsafe.Slice((*uint32)(unsafe.Pointer(&p.mposIn.Bytes()[0])), qwenWide*4)
	for c, x := range xs {
		if at[c].Pos >= s.MaxContext {
			return nil, fmt.Errorf("vk: position %d is past the %d the pipeline was built for", at[c].Pos, s.MaxContext)
		}
		// The cache index is what must be consecutive. The rotation's axes
		// need not be, and an image is exactly the case where they are not.
		if c > 0 && at[c].Pos != at[c-1].Pos+1 {
			return nil, fmt.Errorf("vk: a pass needs consecutive cache positions, given %v", at)
		}
		copy(stream[c*s.Dim:(c+1)*s.Dim], x)
		pos[c] = uint32(at[c].Pos)
		mpos[4*c+0] = uint32(at[c].T)
		mpos[4*c+1] = uint32(at[c].H)
		mpos[4*c+2] = uint32(at[c].W)
		mpos[4*c+3] = 0
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

// noSnapshot is the snapAt that never matches a column.
const noSnapshot = ^uint32(0)

// record lays down one token's whole pass. It is separate from Forward so that
// the same sequence can be compiled once and replayed, which is what tells a
// recording's cost apart from the card's.
func (p *QwenPipeline) record(r *Recorder, columns, snapAt int) {
	s := p.shape
	dim := uint32(s.Dim)
	cols := uint32(columns)
	normFirst := normPush{n: dim, flags: normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normAttn := normPush{n: dim, flags: normAdd | normSum | normGain | normFloat | normQuant, eps: s.Eps, scalar: 1}
	normFFN := normAttn
	normFinal := normPush{n: dim, flags: normAdd | normSum | normGain | normFloat, eps: s.Eps, scalar: 1}

	blocks := min(len(p.isSSM), len(p.ffnBlocks))

	tl := p.tl
	if tl != nil {
		tl.Reset(r)
		tl.Stamp(r, "start")
	}
	r.Copy(p.xs, 0, p.xin, s.Dim*columns*4)
	r.Copy(p.posBuf, 0, p.posIn, 4*columns)
	r.Copy(p.mposBuf, 0, p.mposIn, 16*columns)
	r.Barrier()

	for i := 0; i < blocks; i++ {
		push := &normAttn
		if i == 0 {
			push = &normFirst
		}
		r.DispatchColumns(p.setAttnNorms[i], 1, cols, unsafe.Pointer(push))
		r.Barrier()
		tl.Stamp(r, "norm")

		if p.isSSM[i] {
			p.recordSSM(r, p.ssmBlocks[i], columns, snapAt)
		} else {
			p.recordAttn(r, p.attnBlocks[i], columns)
		}
		r.Barrier()
		if p.isSSM[i] {
			tl.Stamp(r, "ssm out")
		} else {
			tl.Stamp(r, "attn out")
		}

		r.DispatchColumns(p.setFFNNorms[i], 1, cols, unsafe.Pointer(&normFFN))
		r.Barrier()
		tl.Stamp(r, "norm")

		p.recordFFN(r, p.ffnBlocks[i], columns)
		r.Barrier()
		tl.Stamp(r, "ffn down")
	}

	r.DispatchColumns(p.setFinalNorm, 1, cols, unsafe.Pointer(&normFinal))
	r.Barrier()
	tl.Stamp(r, "final norm")
	r.Copy(p.hidden, 0, p.xs, s.Dim*columns*4)
}

// product dispatches one projection at the width of the pass.
//
// A binary answers exactly the number of columns it was built for, so a pass
// wider than the widest binary a pipeline has is run as several dispatches at
// an offset — a hundred and twenty-eight columns through a sixteen-wide
// mat-vec is eight of them, each told where its columns start. The pipelines
// that carry the tiled product have a binary at every width a pass takes and
// never split.
//
// The two push blocks put that offset in different places, so there is one of
// these for each rather than an unsafe.Pointer and a field index.
func (p *QwenPipeline) product(r *Recorder, set *Set, rows, columns int, push moePush) {
	// One slice of the shared dimension, always. The tiled product divides by
	// this to find its slice, and a zero here is a division by zero in the
	// shader — which does not fail, it answers: every wide pass was wrong
	// until this line, and nothing caught it because no test read a prompt
	// past sixteen positions. TestVulkanWidePassMatchesTokenPath does now.
	push.split = 1
	for at := 0; at < columns; {
		// The widest this pipeline has, uncapped: these are the projections
		// that carry the tiled product, and narrowChunk is the mat-vec's
		// ceiling rather than everyone's. Capping here as well left the tiled
		// binaries bound and never dispatched — dispatchAt only reaches them
		// at thirty-two columns or more — so a pass of a hundred and
		// twenty-eight was eight passes of sixteen and the table flattened.
		w := set.widest(columns - at)
		if w == 0 {
			w = 1
		}
		push.col = uint32(at)
		p.dispatchAt(r, set, rows, w, true, unsafe.Pointer(&push))
		at += w
	}
}

// productK is product for the kernels that take vk/ssm.go's push block: the
// K-quant, Q4_1 and float projections, which have no tiled form and are always
// the mat-vec.
func (p *QwenPipeline) productK(r *Recorder, set *Set, rows, columns int, push matvecKPush) {
	for at := 0; at < columns; {
		w := set.widest(min(columns-at, narrowChunk))
		if w == 0 {
			w = 1
		}
		push.Col = uint32(at)
		p.dispatchAt(r, set, rows, w, false, unsafe.Pointer(&push))
		at += w
	}
}

// dispatchAt issues one of those, with the workgroup count the binary bound at
// that width wants: a mat-vec answers a fixed number of outputs to a
// workgroup, and the tiled product answers a tile of rows by a block of
// columns. vk/matmul.go owns both counts.
//
// tiled says which is bound, and it is the caller's to know rather than
// something to infer from the width. Inferring it is a fault this had: the
// tiled count was taken for any pass of thirty-two columns or more, so a
// mat-vec pipeline given a thirty-two-wide binary was dispatched over a
// sixty-fourth of the workgroups it needed. It covered a fraction of the rows,
// left the rest untouched, and measured a third faster for it — which is how a
// wrong kernel looks like a fast one.
func (p *QwenPipeline) dispatchAt(r *Recorder, set *Set, rows, columns int, tiled bool, push unsafe.Pointer) {
	if tiled && columns >= tiledColumns {
		r.DispatchWide(set, columns, coopProductGroups(p.coop, columns, rows), push)
		return
	}
	groups := matvecGroups(rows)
	if columns == 1 {
		r.Dispatch(set, groups, push)
		return
	}
	r.DispatchWide(set, columns, groups, push)
}

// Profile installs a timeline, or takes one out, and forgets the recordings
// so that the next pass is recorded with the stamps in it.
func (p *QwenPipeline) Profile(t *Timeline) {
	p.tl = t
	p.pass = nil
}

// NewTimeline is a timeline sized for one pass of this pipeline: nine stamps a
// block and a few for the ends.
func (p *QwenPipeline) NewTimeline() (*Timeline, error) {
	return p.d.NewTimeline(12*len(p.isSSM) + 8)
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
		r.Copy(p.mposBuf, 0, p.mposIn, 16)
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
		r.Copy(p.mposBuf, 0, p.mposIn, 16)
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
		// Whichever mixer this block has wrote into the one scratch.
		r.Copy(p.hidden, 0, p.mixOut, s.Dim*4)
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
		r.Copy(p.mposBuf, 0, p.mposIn, 16)
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
		switch what {
		case "qkv":
			pick, n = p.qkvBuf, s.ConvDim
		case "gate":
			pick, n = p.gateZBuf, s.Inner
		case "alpha":
			pick, n = p.alphaBuf, s.Rank
		case "beta":
			pick, n = p.betaBuf, s.Rank
		case "conv":
			pick, n = p.convOut, s.ConvDim
		case "y":
			pick, n = p.ySSM, s.Inner
		case "out":
			pick, n = p.mixOut, s.Dim
		}
	} else {
		switch what {
		case "q":
			pick, n = p.qIn, s.qFullDim()
		case "k":
			pick, n = p.kIn, s.kvDim()
		case "v":
			pick, n = p.vIn, s.kvDim()
		case "qrope":
			pick, n = p.qOut, s.qDim()
		case "mix":
			pick, n = p.attnOut, s.qDim()
		case "out":
			pick, n = p.mixOut, s.Dim
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
		r.Copy(p.mposBuf, 0, p.mposIn, 16)
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

// setPos writes the single column of a one-token pass, on every axis at once.
// It is the probes' and the prediction block's entry, and all of them are text:
// a draft sits at the position it drafts for, and a probe replays a text token.
// setPlace is what an image would use.
func (p *QwenPipeline) setPos(pos int) {
	p.setPlace(QwenPlace{Pos: pos, T: pos, H: pos, W: pos})
}

// setPlace writes one column's cache index and its three rotation axes. Both
// buffers are written together on purpose: a pass that set one and not the
// other would rotate by whatever the pass before it left, which is a wrong
// answer that nothing would report.
func (p *QwenPipeline) setPlace(at QwenPlace) {
	w := unsafe.Slice((*uint32)(unsafe.Pointer(&p.posIn.Bytes()[0])), qwenWide)
	w[0] = uint32(at.Pos)
	m := unsafe.Slice((*uint32)(unsafe.Pointer(&p.mposIn.Bytes()[0])), qwenWide*4)
	m[0], m[1], m[2], m[3] = uint32(at.T), uint32(at.H), uint32(at.W), 0
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

	p.product(r, b.setQKV, s.ConvDim, columns, qkv)
	p.product(r, b.setGate, s.Inner, columns, gate)
	p.productK(r, b.setAlpha, s.Rank, columns, small)
	p.productK(r, b.setBeta, s.Rank, columns, small)
	r.Barrier()
	p.tl.Stamp(r, "ssm in")

	// The convolution and the scan carry the columns inside themselves: both
	// hold state that runs from one token to the next, so a column cannot
	// start before the one before it has finished.
	r.Dispatch(b.setConv, uint32((s.ConvDim+255)/256), unsafe.Pointer(&conv))
	r.Barrier()
	p.tl.Stamp(r, "ssm conv")

	// The two norms the recurrence used to carry, before and after it. They are
	// what kept it to one workgroup a head; see shaders/ssm_scan.comp.
	qkn := ssmQKNormPush{ConvDim: uint32(s.ConvDim), Eps: s.Eps}
	r.DispatchColumns(p.setQKNorm, uint32(s.qkHeads()), uint32(columns), unsafe.Pointer(&qkn))
	r.Barrier()
	p.tl.Stamp(r, "ssm qknorm")

	// One workgroup a head and COLS state columns of it: 48 by 16 rather than
	// the 48 this kernel ran in for the life of the engine.
	r.DispatchColumns(b.setScan, uint32(s.Rank), uint32(s.StateSize/scanColumns), unsafe.Pointer(&scan))
	r.Barrier()
	p.tl.Stamp(r, "ssm scan")

	gateNorm := ssmGatePush{Inner: uint32(s.Inner), Eps: s.Eps}
	r.DispatchColumns(b.setNormGate, uint32(s.Rank), uint32(columns), unsafe.Pointer(&gateNorm))
	r.Barrier()
	p.tl.Stamp(r, "ssm gate")

	// Wide enough and the weights Q5_K, and the output projection is the tiled
	// product against the eight-bit form; otherwise the mat-vec against the
	// floats. llama.cpp's own profiler puts this projection and the Q4_1 down
	// at 24ms of a 512-token prompt where ours took 324, and it is the last of
	// the model's matrices to be read sixteen columns at a time.
	// q5kCols and not tiledColumns: the tile answers that many columns and a
	// pass narrower than one would dispatch no workgroups at all and leave the
	// projection undone. Every width qwenWidths names from sixty-four up is a
	// multiple of it.
	if b.setOutWide != nil && columns >= q5kCols && columns%q5kCols == 0 {
		quant := swigluPush{N: uint32(s.Inner), Columns: uint32(columns)}
		r.Dispatch(p.setQuantY, uint32((s.Inner/quantBlock*columns+255)/256), unsafe.Pointer(&quant))
		r.Barrier()
		p.product(r, b.setOutWide, s.Dim, columns,
			moePush{dim: uint32(s.Dim), ffn: uint32(s.Inner), used: 1})
	} else {
		p.productK(r, b.setOut, s.Dim, columns, out)
	}
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
		Sect0:      uint32(s.RoPESections[0]),
		Sect1:      uint32(s.RoPESections[1]),
		Sect2:      uint32(s.RoPESections[2]),
		Sect3:      uint32(s.RoPESections[3]),
	}
	gqa := attnGQAPush{
		MaxContext: uint32(s.MaxContext),
		HeadsPerKV: uint32(s.Heads / s.KVHeads),
		Heads:      uint32(s.Heads),
		Scale:      float32(1 / sqrtOf(s.HeadDim)),
		Columns:    uint32(columns),
	}

	p.product(r, b.setQ, s.qFullDim(), columns, q)
	p.product(r, b.setK, s.kvDim(), columns, kv)
	p.product(r, b.setV, s.kvDim(), columns, kv)
	r.Barrier()
	p.tl.Stamp(r, "attn qkv")

	// Every column's keys and values go into the cache before any column reads
	// them, which is why this is a dispatch of its own and not the head of the
	// one below: a barrier inside a workgroup does not order two workgroups.
	r.DispatchColumns(b.setPrep, uint32(s.Heads+s.KVHeads), uint32(columns), unsafe.Pointer(&prep))
	r.Barrier()
	p.tl.Stamp(r, "attn prep")

	r.DispatchColumns(b.setGQA, uint32(s.Heads), uint32((columns+qAttnTile-1)/qAttnTile), unsafe.Pointer(&gqa))
	r.Barrier()
	p.tl.Stamp(r, "attn gqa")

	if columns >= tiledColumns {
		quant := swigluPush{N: uint32(s.qDim()), Columns: uint32(columns)}
		r.Dispatch(p.setQuantAttn, uint32((s.qDim()/quantBlock*columns+255)/256), unsafe.Pointer(&quant))
		r.Barrier()
		p.product(r, b.setOWide, s.Dim, columns,
			moePush{dim: uint32(s.Dim), ffn: uint32(s.qDim()), used: 1})
	} else {
		p.productK(r, b.setO, s.Dim, columns, out)
	}
}

func (p *QwenPipeline) recordFFN(r *Recorder, b *qwenFFNBlock, columns int) {
	s := p.shape
	up := moePush{dim: uint32(s.FFN), ffn: uint32(s.Dim), used: 1}
	// The activation is elementwise and the columns lie end to end, so a wide
	// pass is the same kernel over more of the buffer.
	act := swigluPush{N: uint32(s.FFN), Columns: uint32(columns)}
	down := matvecKPush{Dim: uint32(s.Dim), FFN: uint32(s.FFN)}

	p.product(r, b.setGate, s.FFN, columns, up)
	p.product(r, b.setUp, s.FFN, columns, up)
	r.Barrier()
	p.tl.Stamp(r, "ffn up")

	// A thread a Q8_0 block, because that is what it quantizes.
	blocks := s.FFN / quantBlock * columns
	r.Dispatch(p.setAct, uint32((blocks+255)/256), unsafe.Pointer(&act))
	r.Barrier()
	p.tl.Stamp(r, "ffn act")

	// Wide enough and the down projection is a tiled product against the
	// eight-bit form; narrow and it is the mat-vec against the floats, which
	// is what a token has always run.
	// Wide enough and the weights Q4_0, and the down projection is the tiled
	// product against the eight-bit form; otherwise the mat-vec against the
	// floats, which is what a token has always run. llama.cpp's own profiler
	// says this projection is the largest matrix product in the model — 95 of
	// its 421ms for a 512-token prompt — so it is the one that had to move.
	switch {
	case b.q41 && columns >= q5kCols && columns%q5kCols == 0:
		// The tiled Q4_1, which is the Q5_K tile over a simpler weight.
		tile := moePush{dim: uint32(s.Dim), ffn: uint32(s.FFN), used: 1}
		rows := (s.Dim + q5kRows - 1) / q5kRows
		r.Dispatch(b.setDownWide, uint32(rows*(columns/q5kCols)), unsafe.Pointer(&tile))
	case !b.q41 && columns >= tiledColumns:
		p.product(r, b.setDownWide, s.Dim, columns,
			moePush{dim: uint32(s.Dim), ffn: uint32(s.FFN), used: 1})
	default:
		p.productK(r, b.setDown, s.Dim, columns, down)
	}
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
	if p.mposIn != nil {
		p.mposIn.Close()
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

// The multi-token-prediction block, which is a Qwen3.8 block with a two-input
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
	if b.setFFNNorm, err = p.pipeNorm.NewSet([]*Buffer{p.xs, p.mixOut, b.ffnNorm, p.none, p.resid, p.ffnNorm, p.ffnNormQ, p.ffnNormS}); err != nil {
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
// DraftMTP drafts at a text position. DraftMTPAt is the same for a draft whose
// axes do not follow the cache index.
func (p *QwenPipeline) DraftMTP(eh []float32, pos int) ([]float32, error) {
	return p.DraftMTPAt(eh, QwenPlace{Pos: pos, T: pos, H: pos, W: pos})
}

func (p *QwenPipeline) DraftMTPAt(eh []float32, at QwenPlace) ([]float32, error) {
	pos := at.Pos
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
	p.setPlace(at)

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
	r.Copy(p.mposBuf, 0, p.mposIn, 16)
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
