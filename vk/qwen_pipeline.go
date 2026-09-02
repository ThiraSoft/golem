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

	"github.com/ThiraSoft/golem/nn"
)

//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q40.comp -o shaders/matvec_q40.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q4k.comp -o shaders/matvec_q4k.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q6k.comp -o shaders/matvec_q6k.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/swiglu_act.comp -o shaders/swiglu_act.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/quant_q80.comp -o shaders/quant_q80.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/qwen_attn_prep.comp -o shaders/qwen_attn_prep.spv
//go:generate glslc -O --target-env=vulkan1.1 -fshader-stage=compute shaders/qwen_attn_gqa.comp -o shaders/qwen_attn_gqa.spv

//go:embed shaders/matvec_q40.spv
var matvecQ40SPIRV []byte

// The Q4_K mat-vec. It was compiled and embedded next to the delta net's
// kernels for a year and bound to nothing, against a layout the split now
// replaces; vk/shaders/matvec_q4k.comp says what it reads.
//
//go:embed shaders/matvec_q4k.spv
var matvecQ4KSPIRV []byte

// And the Q6_K one, which had no kernel at all in a projection: q6k.comp is the
// logit head's, against an activation nothing on this card produces.
//
//go:embed shaders/matvec_q6k.spv
var matvecQ6KSPIRV []byte

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
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q4k.comp -o shaders/matvec_q4k_2.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q6k.comp -o shaders/matvec_q6k_2.spv
//go:generate glslc -O -DCOLUMNS=2 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32_2.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec4.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q4k.comp -o shaders/matvec_q4k_4.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q6k.comp -o shaders/matvec_q6k_4.spv
//go:generate glslc -O -DCOLUMNS=4 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32_4.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q40.comp -o shaders/matvec_q40_8.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q4k.comp -o shaders/matvec_q4k_8.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q6k.comp -o shaders/matvec_q6k_8.spv
//go:generate glslc -O -DCOLUMNS=8 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32_8.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec.comp -o shaders/matvec_qwen16.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q4k.comp -o shaders/matvec_q4k_16.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_q6k.comp -o shaders/matvec_q6k_16.spv
//go:generate glslc -O -DCOLUMNS=16 --target-env=vulkan1.1 -fshader-stage=compute shaders/matvec_f32.comp -o shaders/matvec_f32_16.spv

// The tiled product again, for the Q4_1 weights a Q4_0 checkpoint still keeps:
// llama.cpp's quantizer leaves ffn_down at Q4_1, and it is the largest matrix
// in a block. shaders/matmul.comp's -DQ41 says what differs — twenty bytes a
// block instead of eighteen, and a minimum where the nibble's offset of eight
// was.
//

//go:embed shaders/matvec2.spv
var matvec2SPIRV []byte

//go:embed shaders/matvec_q4k_2.spv
var matvecQ4K_2SPIRV []byte

//go:embed shaders/matvec_q6k_2.spv
var matvecQ6K_2SPIRV []byte

//go:embed shaders/matvec_f32_2.spv
var matvecF32_2SPIRV []byte

//go:embed shaders/matvec4.spv
var matvec4SPIRV []byte

//go:embed shaders/matvec_q4k_4.spv
var matvecQ4K_4SPIRV []byte

//go:embed shaders/matvec_q6k_4.spv
var matvecQ6K_4SPIRV []byte

//go:embed shaders/matvec_f32_4.spv
var matvecF32_4SPIRV []byte

//go:embed shaders/matvec_q40_8.spv
var matvecQ40_8SPIRV []byte

//go:embed shaders/matvec_q4k_8.spv
var matvecQ4K_8SPIRV []byte

//go:embed shaders/matvec_q6k_8.spv
var matvecQ6K_8SPIRV []byte

//go:embed shaders/matvec_f32_8.spv
var matvecF32_8SPIRV []byte

//go:embed shaders/matvec_qwen16.spv
var matvecQwen16SPIRV []byte

//go:embed shaders/matvec_q4k_16.spv
var matvecQ4K_16SPIRV []byte

//go:embed shaders/matvec_q6k_16.spv
var matvecQ6K_16SPIRV []byte

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

//go:embed shaders/matmul_coop_q6k64.spv
var matmulCoopQ6K64SPIRV []byte

//go:embed shaders/matmul_coop_q6k128.spv
var matmulCoopQ6K128SPIRV []byte

//go:embed shaders/matmul_coop_q6k256.spv
var matmulCoopQ6K256SPIRV []byte

//go:embed shaders/matmul_coop_q6k512.spv
var matmulCoopQ6K512SPIRV []byte

//go:embed shaders/matmul_coop_q4k64.spv
var matmulCoopQ4K64SPIRV []byte

//go:embed shaders/matmul_coop_q4k128.spv
var matmulCoopQ4K128SPIRV []byte

//go:embed shaders/matmul_coop_q4k256.spv
var matmulCoopQ4K256SPIRV []byte

//go:embed shaders/matmul_coop_q4k512.spv
var matmulCoopQ4K512SPIRV []byte

//go:embed shaders/matmul_coop_q5k64.spv
var matmulCoopQ5K64SPIRV []byte

//go:embed shaders/matmul_coop_q5k128.spv
var matmulCoopQ5K128SPIRV []byte

//go:embed shaders/matmul_coop_q5k256.spv
var matmulCoopQ5K256SPIRV []byte

//go:embed shaders/matmul_coop_q5k512.spv
var matmulCoopQ5K512SPIRV []byte

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

	// Golem is the format of a .golem checkpoint, and the zero value for one of
	// llama.cpp's own types. It decides which of two forms every projection in
	// the model takes; vk/qwen_golem.go is the other one.
	//
	// A code width would not answer it any more: the trellis has no lattice
	// code, so the question is which format and not how wide.
	Golem nn.Quant

	// Float says every projection arrives as float32 and is read by
	// shaders/matvec_f32.comp, which is the same y = W·x the delta net's two
	// 48-wide projections have always run — only over the whole model.
	//
	// This is not an inference form. Nobody would run a model at four bytes a
	// weight on a card that reads it at half of one. It exists because a
	// conversion has to calibrate the checkpoint it is converting, and a BF16
	// checkpoint is exactly what a conversion reads: the alternative was to
	// measure a quantized twin and hope the salience did not notice, which is
	// a claim nobody here has measured. A BF16 matrix is widened to float on
	// the way up — the shift the host already does everywhere else — so no
	// shader knows this format exists.
	//
	// It costs four bytes a weight in device memory, which is why it only ever
	// makes sense beside a window: see AddBlockWindow.
	Float bool
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
	// PreQKV is the vector the four input projections read their activation
	// through, and PreO the one the output projection reads. Both nil for a
	// checkpoint that is not .golem; nn.GolemVectorNames says where they come
	// from and cmd/golemquant writes them.
	PreQKV []float32
	PreO   []float32

	WQKV   []byte
	WGate  []byte
	WAlpha []byte
	WBeta  []byte
	WOut   []byte
	// Out is what WOut is stored as. It is a type and not a pair of booleans
	// because the pair had a default, and the default was wrong: a Q4_K_M
	// checkpoint keeps this projection in Q4_0, matched neither flag, and was
	// read as the Q4_1 the default assumed — 20 bytes a block against 18. It
	// happened to run off the end of the tensor and panic. Had the shapes
	// allowed it, the projection would have decoded as noise with nothing to
	// say so.
	//
	// A form this pipeline has no kernel for is an error naming it, never a
	// guess.
	Out nn.Quant
	// QKV and Gate are what the two input projections are stored as, for the
	// same reason and with QwenAttnData.Formats' stake: both were uploaded
	// through splitQ4_0 with nothing asked, and a Q4_K row has Q4_0's length.
	QKV  nn.Quant
	Gate nn.Quant

	ConvWeight []float32
	SSMA       []float32
	SSMDtBias  []float32
	SSMNorm    []float32
}

type QwenAttnData struct {
	// PreQKV and PreO are the two sites' vectors; see QwenSSMData.
	PreQKV []float32
	PreO   []float32

	WQ    []byte
	WK    []byte
	WV    []byte
	WO    []byte
	QNorm []float32
	KNorm []float32
	// Formats is what those four are stored as, for the reason
	// QwenSSMData.Out gives and with more at stake: the four were uploaded
	// through splitQ4_0 with nothing asked at all, and a Q4_K row is eighteen
	// bytes to a block of thirty-two exactly as a Q4_0 row is — so a K-quant
	// attention read as Q4_0 does not overrun anything. It answers, and what
	// it answers is noise.
	//
	// The zero value is F32 and a float checkpoint is the shape's business, so
	// a caller who fills the matrices and forgets this is refused by name.
	Formats BlockFormats
}

type QwenFFNData struct {
	// PreGateUp and PreDown are the two sites' vectors; see QwenSSMData.
	PreGateUp []float32
	PreDown   []float32

	Gate []byte
	Up   []byte
	Down []byte
	// GateUp and DownQ are what those matrices are stored as. Types and not
	// booleans, for the reason QwenSSMData.Out gives: a boolean has a default,
	// and a default is a guess about somebody else's file. The gate and the up
	// were uploaded as Q4_0 whatever they were, so a Q3_K_M checkpoint — whose
	// blocks are 13.75 bytes to Q4_0's 18 — ran off the end of the tensor and
	// panicked with a slice bound. That is the failure that looks like a bug in
	// the reader rather than a form it does not support.
	GateUp nn.Quant
	DownQ  nn.Quant
}

type qwenSSMBlock struct {
	// index is which block of the trunk this is, and -1 for the prediction
	// block's copies, which belong to no position in it. A calibration files
	// its accumulators under it.
	index int

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
	// f32 is qwenFFNBlock's.
	f32 bool
	// golem is the five projections in their .golem form, and nil for a
	// checkpoint of any other type. Everything above it that is not a
	// projection — the convolution, the recurrence, the two norms — is the
	// same either way and is used by both.
	golem *qwenGolemSSM
}

type qwenAttnBlock struct {
	index int // see qwenSSMBlock

	wQ, wK, wV, wO *Buffer
	qNorm, kNorm   *Buffer
	kCache, vCache *Buffer

	setQ, setK, setV *Set
	setPrep, setGQA  *Set
	setO             *Set
	// f32 is qwenFFNBlock's.
	f32 bool
	// golem is the four projections in their .golem form; see qwenSSMBlock.
	golem *qwenGolemAttn
}

type qwenFFNBlock struct {
	index int // see qwenSSMBlock

	wGate, wUp, wDown       *Buffer
	setGate, setUp, setDown *Set
	// f32 says the three projections are float32 and read by pipeMatF32. It is
	// the shape's Float, kept on the block so that record does not reach for
	// the shape to answer a question about the matrices in front of it.
	//
	// It is the only thing a block still says about its own weights. Every
	// quantized form is a pipeline vk/quantproduct.go handed out at upload,
	// and the dispatch that reads it is the same one whichever it was.
	f32 bool
	// golem is the three projections in their .golem form, and nil for a
	// checkpoint of any other type. See qwenSSMBlock.
	golem *qwenGolemFFN
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

	pipeNorm *Pipeline
	// The two K-quant mat-vecs against *float* activations, which the
	// prediction block's front projection is the last reader of: it reads two
	// hidden states joined, and nothing quantizes those.
	pipeMatQ4K   *Pipeline
	pipeMatQ6K   *Pipeline
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
	// quants is one pipeline per K-quant weight format, built as a block asks
	// for it. vk/quantproduct.go owns it, and it is what vk/attention.go and
	// vk/mixture.go read their K-quants through: the same four buffers and the
	// same push block whatever the format, so a caller binds what it always
	// bound and only the unpacking inside the shader differs.
	quants *quantProducts

	// The lattice and the transform, for a .golem checkpoint. Both belong to
	// the device rather than to a matrix — one table and two pipelines serve
	// every site of every block — and both are nil for any other type.
	golem *GolemKernels
	preps *GolemPrepares

	// calib is the per-site accumulators, when a conversion is measuring what
	// each matrix is fed. Nil the rest of the time, and every dispatch it
	// would have added is nil too.
	calib *qwenCalib
	// blockBase is which block of the model this pipeline's first block is.
	// Zero for a pipeline holding the whole trunk, and the window's first
	// block when a calibration is streaming the model past the card a window
	// at a time: the accumulators are filed under the model's numbering, not
	// the window's, or every window would answer for block zero.
	blockBase int
	// noHead drops the head's accumulator. Only the window that holds the last
	// block sees what the final norm really makes; every earlier one runs that
	// norm over a state the rest of the model has not finished with.
	noHead bool

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
	p.quants = newQuantProducts(d, p.coop)
	var err error

	type build struct {
		into   **Pipeline
		spirv  []byte
		binds  int
		pushSz uintptr
	}
	for _, b := range []build{
		{&p.pipeNorm, normWideSPIRV, 8, unsafe.Sizeof(normPush{})},
		{&p.pipeMatQ4K, matvecQ4KSPIRV, 3, unsafe.Sizeof(matvecKPush{})},
		{&p.pipeMatQ6K, matvecQ6KSPIRV, 3, unsafe.Sizeof(matvecKPush{})},
		{&p.pipeMatF32, matvecF32SPIRV, 3, unsafe.Sizeof(matvecKPush{})},
		{&p.pipeSwiglu, swigluActSPIRV, 5, unsafe.Sizeof(swigluPush{})},
		{&p.pipeQuant, quantQ80SPIRV, 3, unsafe.Sizeof(swigluPush{})},
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
	// The two K-quant mat-vecs against floats, at the widths the prediction
	// block might take, and the float mat-vec at all of them. Everything else
	// a block reads is a vk/quantproduct.go pipeline, built when a matrix in
	// that format is first uploaded and carrying its own ten binaries.
	for _, w := range []struct {
		pipe    *Pipeline
		columns int
		spirv   []byte
	}{
		{p.pipeMatQ4K, 2, matvecQ4K_2SPIRV}, {p.pipeMatQ4K, 4, matvecQ4K_4SPIRV}, {p.pipeMatQ4K, 8, matvecQ4K_8SPIRV}, {p.pipeMatQ4K, 16, matvecQ4K_16SPIRV},
		{p.pipeMatQ6K, 2, matvecQ6K_2SPIRV}, {p.pipeMatQ6K, 4, matvecQ6K_4SPIRV}, {p.pipeMatQ6K, 8, matvecQ6K_8SPIRV}, {p.pipeMatQ6K, 16, matvecQ6K_16SPIRV},
		{p.pipeMatF32, 2, matvecF32_2SPIRV}, {p.pipeMatF32, 4, matvecF32_4SPIRV}, {p.pipeMatF32, 8, matvecF32_8SPIRV}, {p.pipeMatF32, 16, matvecF32_16SPIRV},
	} {
		if err := w.pipe.Wide(w.columns, w.spirv); err != nil {
			return nil, err
		}
	}

	// A .golem checkpoint: one lattice table and two transform pipelines for
	// the whole model, whatever any block does with them.
	if shape.Golem.Golem() {
		if p.golem, err = NewGolemKernels(d, shape.Golem); err != nil {
			return nil, err
		}
		if p.preps, err = NewGolemPrepares(d); err != nil {
			p.Close()
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

// uploadQuant puts one quantized projection on the card in the layout its
// kernels read, and hands back the pipeline that reads it.
//
// Every quantized projection in this pipeline goes through here, whatever its
// format, and reads the *eight-bit* form of whatever feeds it. Both halves of
// that were once otherwise. Each format had a mat-vec of its own against float
// activations and a tiled form only some of them had, so a block carried a
// pipeline per matrix and a boolean per decision — and the booleans had
// defaults, which are guesses about somebody else's file. Two of them guessed
// wrong on a checkpoint this repository had never opened.
//
// A length is never what says which format a matrix is: eighteen bytes to a
// block of thirty-two is Q4_0 and it is also Q4_K, so quantLayout is given the
// type the file declared and nothing is inferred.
// uploadInto is uploadQuant writing through a pointer, which is what the
// tables below want: one row of a struct literal a matrix, and the loop reads
// like the list of projections it is.
func (p *QwenPipeline) uploadInto(into **Buffer, data []byte, rows, cols int, q nn.Quant) (*Pipeline, error) {
	buf, pipe, err := p.uploadQuant(data, rows, cols, q)
	if err != nil {
		return nil, err
	}
	*into = buf
	return pipe, nil
}

func (p *QwenPipeline) uploadQuant(data []byte, rows, cols int, q nn.Quant) (*Buffer, *Pipeline, error) {
	layout, err := quantLayout(q, data, rows, cols)
	if err != nil {
		return nil, nil, err
	}
	pipe, err := p.quants.get(q)
	if err != nil {
		return nil, nil, err
	}
	buf, err := p.upload(layout)
	if err != nil {
		return nil, nil, err
	}
	return buf, pipe, nil
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
	b := &qwenSSMBlock{index: i}
	var err error

	if !p.usesGolem() {
		if err := p.uploadSSMProjections(b, d); err != nil {
			return err
		}
	}
	for _, u := range []struct {
		into **Buffer
		data []byte
	}{
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
	if b.setConv, err = p.pipeConv.NewSet([]*Buffer{b.convWeight, p.qkvBuf, b.convState, p.convOut, b.shadowConv}); err != nil {
		return err
	}
	if b.setScan, err = p.pipeScan.NewSet([]*Buffer{p.convOut, p.qkNorm, p.alphaBuf, b.dtBias, b.ssmA, p.betaBuf, b.ssmState, p.ySSM, b.shadowState}); err != nil {
		return err
	}
	if b.setNormGate, err = p.pipeGate.NewSet([]*Buffer{p.ySSM, p.gateZBuf, b.ssmNorm}); err != nil {
		return err
	}
	if p.usesGolem() {
		if b.golem, err = p.newGolemSSM(b, d); err != nil {
			return err
		}
	}
	p.ssmBlocks[i] = b
	return nil
}

// uploadSSMProjections is a delta net's five weight matrices in one of
// llama.cpp's types, and the descriptors that read them. A .golem checkpoint
// takes vk/qwen_golem.go's path instead and none of this runs.
func (p *QwenPipeline) uploadSSMProjections(b *qwenSSMBlock, d QwenSSMData) error {
	s := p.shape
	var err error
	if s.Float {
		b.f32 = true
	}
	if b.f32 {
		if b.wQKV, err = p.upload(d.WQKV); err != nil {
			return err
		}
		if b.wGate, err = p.upload(d.WGate); err != nil {
			return err
		}
		if b.wOut, err = p.upload(d.WOut); err != nil {
			return err
		}
		if b.setQKV, err = p.pipeMatF32.NewSet([]*Buffer{b.wQKV, p.normed, p.qkvBuf}); err != nil {
			return err
		}
		if b.setGate, err = p.pipeMatF32.NewSet([]*Buffer{b.wGate, p.normed, p.gateZBuf}); err != nil {
			return err
		}
		if b.setOut, err = p.pipeMatF32.NewSet([]*Buffer{b.wOut, p.ySSM, p.mixOut}); err != nil {
			return err
		}
	} else {
		// The two input projections read the norm's eight-bit form and the
		// output projection reads the recurrence's, which recordSSM writes for
		// it. Three matrices, three asks of the same door.
		for _, u := range []struct {
			into       **Buffer
			set        **Set
			xq, xs, y  *Buffer
			data       []byte
			rows, cols int
			q          nn.Quant
			name       string
		}{
			{&b.wQKV, &b.setQKV, p.normedQ, p.normedS, p.qkvBuf, d.WQKV, s.ConvDim, s.Dim, d.QKV, "input"},
			{&b.wGate, &b.setGate, p.normedQ, p.normedS, p.gateZBuf, d.WGate, s.Inner, s.Dim, d.Gate, "gate"},
			{&b.wOut, &b.setOut, p.ySSMQ, p.ySSMS, p.mixOut, d.WOut, s.Dim, s.Inner, d.Out, "output"},
		} {
			pipe, perr := p.uploadInto(u.into, u.data, u.rows, u.cols, u.q)
			if perr != nil {
				return fmt.Errorf("vk: the delta net's %s projection: %w", u.name, perr)
			}
			if *u.set, err = pipe.NewSet([]*Buffer{*u.into, u.xq, u.xs, u.y}); err != nil {
				return err
			}
		}
	}
	// The decay's two projections are floats in every checkpoint — a rank of a
	// few dozen outputs, which nobody has ever thought worth quantizing.
	for _, u := range []struct {
		into **Buffer
		data []byte
	}{
		{&b.wAlpha, d.WAlpha}, {&b.wBeta, d.WBeta},
	} {
		if *u.into, err = p.upload(u.data); err != nil {
			return err
		}
	}
	if b.setAlpha, err = p.pipeMatF32.NewSet([]*Buffer{b.wAlpha, p.normed, p.alphaBuf}); err != nil {
		return err
	}
	if b.setBeta, err = p.pipeMatF32.NewSet([]*Buffer{b.wBeta, p.normed, p.betaBuf}); err != nil {
		return err
	}
	return nil
}

func (p *QwenPipeline) AddAttnBlock(i int, d QwenAttnData) error {
	b, err := p.newAttnBlock(d)
	if err != nil {
		return err
	}
	b.index = i
	p.attnBlocks[i] = b
	return nil
}

func (p *QwenPipeline) newAttnBlock(d QwenAttnData) (*qwenAttnBlock, error) {
	s := p.shape
	b := &qwenAttnBlock{index: -1}
	var err error

	if p.usesGolem() {
		if b.golem, err = p.newGolemAttn(d); err != nil {
			return nil, err
		}
	} else if s.Float {
		b.f32 = true
		for _, u := range []struct {
			into **Buffer
			data []byte
		}{{&b.wQ, d.WQ}, {&b.wK, d.WK}, {&b.wV, d.WV}, {&b.wO, d.WO}} {
			if *u.into, err = p.upload(u.data); err != nil {
				return nil, err
			}
		}
	} else {
		// The three input projections read the norm's eight-bit form and the
		// output projection reads the mix's, which recordAttn writes for it.
		for _, u := range []struct {
			into       **Buffer
			set        **Set
			xq, xs, y  *Buffer
			data       []byte
			rows, cols int
			q          nn.Quant
			name       string
		}{
			{&b.wQ, &b.setQ, p.normedQ, p.normedS, p.qIn, d.WQ, s.qFullDim(), s.Dim, d.Formats.Q, "query"},
			{&b.wK, &b.setK, p.normedQ, p.normedS, p.kIn, d.WK, s.kvDim(), s.Dim, d.Formats.K, "key"},
			{&b.wV, &b.setV, p.normedQ, p.normedS, p.vIn, d.WV, s.kvDim(), s.Dim, d.Formats.V, "value"},
			{&b.wO, &b.setO, p.attnOutQ, p.attnOutS, p.mixOut, d.WO, s.Dim, s.qDim(), d.Formats.O, "output"},
		} {
			pipe, perr := p.uploadInto(u.into, u.data, u.rows, u.cols, u.q)
			if perr != nil {
				return nil, fmt.Errorf("vk: the attention's %s projection: %w", u.name, perr)
			}
			if *u.set, err = pipe.NewSet([]*Buffer{*u.into, u.xq, u.xs, u.y}); err != nil {
				return nil, err
			}
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
	if b.f32 {
		for _, u := range []struct {
			into **Set
			w, y *Buffer
		}{{&b.setQ, b.wQ, p.qIn}, {&b.setK, b.wK, p.kIn}, {&b.setV, b.wV, p.vIn}} {
			if *u.into, err = p.pipeMatF32.NewSet([]*Buffer{u.w, p.normed, u.y}); err != nil {
				return nil, err
			}
		}
	}
	if b.setPrep, err = p.pipeAttnPrep.NewSet([]*Buffer{p.qIn, p.kIn, p.vIn, b.qNorm, b.kNorm, p.qOut, b.kCache, b.vCache, p.posBuf, p.mposBuf}); err != nil {
		return nil, err
	}
	if b.setGQA, err = p.pipeAttnGQA.NewSet([]*Buffer{p.qOut, b.kCache, b.vCache, p.qIn, p.attnOut, p.posBuf}); err != nil {
		return nil, err
	}
	if b.f32 {
		// The mix arrives in floats and stays there: nothing quantizes it on
		// the float path, so this projection reads what the attention wrote.
		if b.setO, err = p.pipeMatF32.NewSet([]*Buffer{b.wO, p.attnOut, p.mixOut}); err != nil {
			return nil, err
		}
	}

	return b, nil
}

func (p *QwenPipeline) AddFFNBlock(d QwenFFNData) error {
	b, err := p.newFFNBlock(d)
	if err != nil {
		return err
	}
	b.index = len(p.ffnBlocks)
	p.ffnBlocks = append(p.ffnBlocks, b)
	return nil
}

func (p *QwenPipeline) newFFNBlock(d QwenFFNData) (*qwenFFNBlock, error) {
	s := p.shape
	b := &qwenFFNBlock{index: -1}
	var err error
	if p.usesGolem() {
		if b.golem, err = p.newGolemFFN(d); err != nil {
			return nil, err
		}
		return b, nil
	}
	if s.Float {
		b.f32 = true
		for _, u := range []struct {
			into **Buffer
			data []byte
		}{{&b.wGate, d.Gate}, {&b.wUp, d.Up}, {&b.wDown, d.Down}} {
			if *u.into, err = p.upload(u.data); err != nil {
				return nil, err
			}
		}
		// Against the float form of what feeds each: the norm writes both its
		// float and its Q8_0 output, and the activation between the gate and
		// the down projection likewise, so this reads what is already there.
		for _, u := range []struct {
			into    **Set
			w, x, y *Buffer
		}{
			{&b.setGate, b.wGate, p.ffnNorm, p.gateBuf},
			{&b.setUp, b.wUp, p.ffnNorm, p.upBuf},
			{&b.setDown, b.wDown, p.actBuf, p.ffnOut},
		} {
			if *u.into, err = p.pipeMatF32.NewSet([]*Buffer{u.w, u.x, u.y}); err != nil {
				return nil, err
			}
		}
		return b, nil
	}
	// Three matrices, three asks of the same door. The gate and the up read
	// the norm's eight-bit form and the down reads the activation's; both are
	// written by the kernel in front of them whatever reads them next, so
	// there is nothing to arrange and no width to decide.
	for _, u := range []struct {
		into       **Buffer
		set        **Set
		xq, xs, y  *Buffer
		rows, cols int
		q          nn.Quant
		data       []byte
		name       string
	}{
		{&b.wGate, &b.setGate, p.ffnNormQ, p.ffnNormS, p.gateBuf, s.FFN, s.Dim, d.GateUp, d.Gate, "gate"},
		{&b.wUp, &b.setUp, p.ffnNormQ, p.ffnNormS, p.upBuf, s.FFN, s.Dim, d.GateUp, d.Up, "up"},
		{&b.wDown, &b.setDown, p.actQ, p.actS, p.ffnOut, s.Dim, s.FFN, d.DownQ, d.Down, "down"},
	} {
		pipe, perr := p.uploadInto(u.into, u.data, u.rows, u.cols, u.q)
		if perr != nil {
			return nil, fmt.Errorf("vk: the feed forward's %s projection: %w", u.name, perr)
		}
		if *u.set, err = pipe.NewSet([]*Buffer{*u.into, u.xq, u.xs, u.y}); err != nil {
			return nil, err
		}
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
		p.accumulate(r, i, "qkv", columns)

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
		p.accumulate(r, i, "gateup", columns)

		p.recordFFN(r, p.ffnBlocks[i], columns)
		r.Barrier()
		tl.Stamp(r, "ffn down")
	}

	r.DispatchColumns(p.setFinalNorm, 1, cols, unsafe.Pointer(&normFinal))
	r.Barrier()
	tl.Stamp(r, "final norm")
	p.accumulate(r, -1, "head", columns)
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
	p.forgetPasses()
}

// forgetPasses drops the compiled recordings, so that the next pass is laid
// down again with whatever has just changed about what a pass does.
func (p *QwenPipeline) forgetPasses() {
	for _, prog := range p.pass {
		prog.Close()
	}
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
	if b.golem != nil {
		p.recordGolemSSM(r, b.golem, b, columns, snapAt)
		return
	}
	s := p.shape
	qkv := moePush{dim: uint32(s.ConvDim), ffn: uint32(s.Dim), used: 1}
	gate := moePush{dim: uint32(s.Inner), ffn: uint32(s.Dim), used: 1}
	small := matvecKPush{Dim: uint32(s.Rank), FFN: uint32(s.Dim)}
	out := matvecKPush{Dim: uint32(s.Dim), FFN: uint32(s.Inner)}

	if b.f32 {
		p.productK(r, b.setQKV, s.ConvDim, columns, matvecKPush{Dim: uint32(s.ConvDim), FFN: uint32(s.Dim)})
		p.productK(r, b.setGate, s.Inner, columns, matvecKPush{Dim: uint32(s.Inner), FFN: uint32(s.Dim)})
	} else {
		p.product(r, b.setQKV, s.ConvDim, columns, qkv)
		p.product(r, b.setGate, s.Inner, columns, gate)
	}
	p.productK(r, b.setAlpha, s.Rank, columns, small)
	p.productK(r, b.setBeta, s.Rank, columns, small)
	r.Barrier()
	p.tl.Stamp(r, "ssm in")

	p.recordSSMState(r, b, columns, snapAt)
	p.accumulate(r, b.index, "o", columns)

	if b.f32 {
		p.productK(r, b.setOut, s.Dim, columns, out)
		return
	}
	// The recurrence's output in eight bits, then the projection over it, at
	// every width. This used to be the wide pass's arrangement alone and the
	// narrow one read floats through a mat-vec of the format's own; llama.cpp's
	// own profiler put this projection and the Q4_1 down at 24ms of a
	// 512-token prompt where ours took 324, which is what moving it here was
	// worth. The quantizing dispatch is one kernel over a few thousand values.
	quant := swigluPush{N: uint32(s.Inner), Columns: uint32(columns)}
	r.Dispatch(p.setQuantY, uint32((s.Inner/quantBlock*columns+255)/256), unsafe.Pointer(&quant))
	r.Barrier()
	p.product(r, b.setOut, s.Dim, columns,
		moePush{dim: uint32(s.Dim), ffn: uint32(s.Inner), used: 1})
}

// recordSSMState is everything between a delta net's input projections and its
// output one: the convolution, the two norms and the recurrence. None of them
// reads a quantized weight — a delta net's convolution, its decay and its norms
// are floats in every checkpoint — so all of it is shared with the Golem path.
func (p *QwenPipeline) recordSSMState(r *Recorder, b *qwenSSMBlock, columns, snapAt int) {
	s := p.shape
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
}

func (p *QwenPipeline) recordAttn(r *Recorder, b *qwenAttnBlock, columns int) {
	if b.golem != nil {
		p.recordGolemAttn(r, b.golem, b, columns)
		return
	}
	s := p.shape
	q := moePush{dim: uint32(s.qFullDim()), ffn: uint32(s.Dim), used: 1}
	kv := moePush{dim: uint32(s.kvDim()), ffn: uint32(s.Dim), used: 1}
	out := matvecKPush{Dim: uint32(s.Dim), FFN: uint32(s.qDim())}

	if b.f32 {
		p.productK(r, b.setQ, s.qFullDim(), columns, matvecKPush{Dim: uint32(s.qFullDim()), FFN: uint32(s.Dim)})
		p.productK(r, b.setK, s.kvDim(), columns, matvecKPush{Dim: uint32(s.kvDim()), FFN: uint32(s.Dim)})
		p.productK(r, b.setV, s.kvDim(), columns, matvecKPush{Dim: uint32(s.kvDim()), FFN: uint32(s.Dim)})
		r.Barrier()
		p.tl.Stamp(r, "attn qkv")
		p.recordAttnMix(r, b, columns)
		p.accumulate(r, b.index, "o", columns)
		p.productK(r, b.setO, s.Dim, columns, out)
		return
	}

	p.product(r, b.setQ, s.qFullDim(), columns, q)
	p.product(r, b.setK, s.kvDim(), columns, kv)
	p.product(r, b.setV, s.kvDim(), columns, kv)
	r.Barrier()
	p.tl.Stamp(r, "attn qkv")

	p.recordAttnMix(r, b, columns)
	p.accumulate(r, b.index, "o", columns)

	// The mix in eight bits, then the output projection over it, at every
	// width — recordSSM says why that is no longer a decision.
	quant := swigluPush{N: uint32(s.qDim()), Columns: uint32(columns)}
	r.Dispatch(p.setQuantAttn, uint32((s.qDim()/quantBlock*columns+255)/256), unsafe.Pointer(&quant))
	r.Barrier()
	p.product(r, b.setO, s.Dim, columns,
		moePush{dim: uint32(s.Dim), ffn: uint32(s.qDim()), used: 1})
}

// recordAttnMix is everything between a full attention's input projections and
// its output one: the rotation and the caches, then the softmax. Neither reads
// a weight matrix, so both are the same kernels over the same buffers whatever
// the projections around them are stored as — which is what lets the Golem path
// in vk/qwen_golem.go be four matrices and two transforms rather than a block.
func (p *QwenPipeline) recordAttnMix(r *Recorder, b *qwenAttnBlock, columns int) {
	s := p.shape
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
	// Every column's keys and values go into the cache before any column reads
	// them, which is why this is a dispatch of its own and not the head of the
	// one below: a barrier inside a workgroup does not order two workgroups.
	r.DispatchColumns(b.setPrep, uint32(s.Heads+s.KVHeads), uint32(columns), unsafe.Pointer(&prep))
	r.Barrier()
	p.tl.Stamp(r, "attn prep")

	r.DispatchColumns(b.setGQA, uint32(s.Heads), uint32((columns+qAttnTile-1)/qAttnTile), unsafe.Pointer(&gqa))
	r.Barrier()
	p.tl.Stamp(r, "attn gqa")
}

func (p *QwenPipeline) recordFFN(r *Recorder, b *qwenFFNBlock, columns int) {
	if b.golem != nil {
		p.recordGolemFFN(r, b.golem, columns)
		return
	}
	s := p.shape
	up := moePush{dim: uint32(s.FFN), ffn: uint32(s.Dim), used: 1}
	// The activation is elementwise and the columns lie end to end, so a wide
	// pass is the same kernel over more of the buffer.
	act := swigluPush{N: uint32(s.FFN), Columns: uint32(columns)}
	down := matvecKPush{Dim: uint32(s.Dim), FFN: uint32(s.FFN)}

	if b.f32 {
		// The float projections have no tiled form: productK is the mat-vec at
		// whichever compiled width the pass reaches, which is what the K-quant
		// and Q4_1 projections have always taken.
		wide := matvecKPush{Dim: uint32(s.FFN), FFN: uint32(s.Dim)}
		p.productK(r, b.setGate, s.FFN, columns, wide)
		p.productK(r, b.setUp, s.FFN, columns, wide)
		r.Barrier()
		p.tl.Stamp(r, "ffn up")
		r.Dispatch(p.setAct, uint32((s.FFN/quantBlock*columns+255)/256), unsafe.Pointer(&act))
		r.Barrier()
		p.tl.Stamp(r, "ffn act")
		p.accumulate(r, b.index, "down", columns)
		p.productK(r, b.setDown, s.Dim, columns, down)
		return
	}

	p.product(r, b.setGate, s.FFN, columns, up)
	p.product(r, b.setUp, s.FFN, columns, up)
	r.Barrier()
	p.tl.Stamp(r, "ffn up")

	// A thread a Q8_0 block, because that is what it quantizes.
	blocks := s.FFN / quantBlock * columns
	r.Dispatch(p.setAct, uint32((blocks+255)/256), unsafe.Pointer(&act))
	r.Barrier()
	p.tl.Stamp(r, "ffn act")
	p.accumulate(r, b.index, "down", columns)

	// The down projection over that eight-bit form, at every width, and it
	// costs nothing to reach: the activation kernel above writes the floats and
	// the Q8_0 pair in the same pass, whoever reads them. llama.cpp's own
	// profiler says this is the largest matrix product in the model — 95 of its
	// 421ms for a 512-token prompt — which is why it was the first to move.
	p.product(r, b.setDown, s.Dim, columns,
		moePush{dim: uint32(s.Dim), ffn: uint32(s.FFN), used: 1})
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
		if p.mtp.golemEH != nil {
			p.mtp.golemEH.Close()
		}
		if p.mtp.preEH != nil {
			p.mtp.preEH.Close()
		}
		for _, b := range []*qwenGolemAttn{mtpAttnGolem(p.mtp)} {
			if b != nil {
				b.Close()
			}
		}
		if p.mtp.ffn != nil && p.mtp.ffn.golem != nil {
			p.mtp.ffn.golem.Close()
		}
		p.mtp.ehIn.Close()
		p.mtp = nil
	}
	// The .golem form of every block, before the buffers they read.
	for _, b := range p.ffnBlocks {
		if b != nil && b.golem != nil {
			b.golem.Close()
		}
	}
	for _, b := range p.attnBlocks {
		if b != nil && b.golem != nil {
			b.golem.Close()
		}
	}
	for _, b := range p.ssmBlocks {
		if b != nil && b.golem != nil {
			b.golem.Close()
		}
	}
	if p.preps != nil {
		p.preps.Close()
		p.preps = nil
	}
	if p.quants != nil {
		p.quants.Close()
		p.quants = nil
	}
	if p.golem != nil {
		p.golem.Close()
		p.golem = nil
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
		p.pipeNorm, p.pipeMatQ4K, p.pipeMatQ6K,
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
	// EHQ is what those bytes are. Q8_0 in the checkpoints seen first, Q4_K in
	// a Q4_K_M, and the two have no length in common, so this is asked rather
	// than assumed.
	EHQ nn.Quant
	// PreEH is the vector that projection reads its input through in a .golem
	// checkpoint. No calibration site names this matrix — the model never runs
	// at it, the prediction block being past the trunk the corpus walks — so
	// what the converter wrote beside it is the blind rotation's own vector,
	// under the tensor's name rather than a site's.
	PreEH []float32
	Attn  QwenAttnData
	FFN   QwenFFNData

	AttnNorm []float32
	FFNNorm  []float32
	HeadNorm []float32
}

type qwenMTPBlock struct {
	// pEH is the pipeline the front projection is read by, one per format.
	pEH   *Pipeline
	wEH   *Buffer
	ehIn  *Buffer
	ehBuf *Buffer

	attn *qwenAttnBlock
	ffn  *qwenFFNBlock

	attnNorm *Buffer
	ffnNorm  *Buffer
	headNorm *Buffer

	setEH *Set
	// The .golem form of that projection: the transform its input goes
	// through, and the matrix. Both nil for a checkpoint of any other type.
	preEH       *PrepareGolem
	golemEH     *GolemMatrix
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

	if !p.usesGolem() && p.pipeMatQ80 == nil {
		if p.pipeMatQ80, err = p.d.NewPipeline(matvecQ80SPIRV, 3, uint32(unsafe.Sizeof(matvecKPush{}))); err != nil {
			return err
		}
	}
	if b.ehIn, err = p.d.Host(s.Dim*2*4, bufferUsageStorage|bufferUsageTransferSrc); err != nil {
		return err
	}
	if b.ehBuf, err = p.local(s.Dim * 2 * 4); err != nil {
		return err
	}
	if p.usesGolem() {
		if len(d.PreEH) != s.Dim*2 {
			return fmt.Errorf("vk: the prediction block's vector is %d wide, its projection reads %d", len(d.PreEH), s.Dim*2)
		}
		if b.preEH, err = p.preps.Bind(b.ehBuf, d.PreEH, prepareGolemGroup); err != nil {
			return err
		}
		if b.golemEH, err = NewGolemMatrixOn(p.golem, d.EHProj, s.Dim, s.Dim*2, b.ehBuf, p.xs); err != nil {
			return err
		}
	} else {
		// Against the floats of the two hidden states joined, which is what
		// every mat-vec of the K-quant tier here also reads: those kernels
		// take vk/ssm.go's push block and three buffers, exactly as the Q8_0
		// one does, so only the pipeline differs.
		switch d.EHQ {
		case nn.Q8_0:
			b.pEH = p.pipeMatQ80
			b.wEH, err = p.upload(d.EHProj)
		case nn.Q4_K:
			b.pEH = p.pipeMatQ4K
			b.wEH, err = p.upload(splitQ4_K(d.EHProj, s.Dim, s.Dim*2))
		case nn.Q6_K:
			b.pEH = p.pipeMatQ6K
			b.wEH, err = p.upload(splitQ6_K(d.EHProj, s.Dim, s.Dim*2))
		default:
			return fmt.Errorf("vk: the prediction block's projection is %s, which this pipeline has no kernel for", d.EHQ)
		}
		if err != nil {
			return err
		}
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

	if !p.usesGolem() {
		if b.setEH, err = b.pEH.NewSet([]*Buffer{b.wEH, b.ehBuf, p.xs}); err != nil {
			return err
		}
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

// mtpAttnGolem is the prediction block's attention in its .golem form, or nil.
// Its blocks are not in the pipeline's maps — it holds its own — so Close has
// to reach them here.
func mtpAttnGolem(b *qwenMTPBlock) *qwenGolemAttn {
	if b.attn == nil {
		return nil
	}
	return b.attn.golem
}

func (p *QwenPipeline) HasMTP() bool { return p.mtp != nil }

// Columns is how many tokens one pass can carry.
func (p *QwenPipeline) Columns() int { return p.widestPass() }

// WidthFor is the widest pass that fits n remaining tokens. A run is read in
// passes of these rather than one width and a ragged tail of single columns:
// the mat-vec binaries exist at four widths and the largest that fits wins.
func (p *QwenPipeline) WidthFor(n int) int {
	for _, w := range qwenWidths {
		if w <= n && w <= p.widestPass() {
			return w
		}
	}
	return 1
}

// golemWidestPass is how wide a pass a .golem model takes.
//
// Not because a wider one is wrong — vk/qwen_golem.go answers any width and
// TestVulkanGolemWidePassMatchesTokenPath holds it to the token path bit for
// bit — but because of what a wide one costs to record. The Golem kernels are
// built for eight columns and no more, where the quantized products have a
// tiled form for a wide pass, so a pass of two hundred and fifty-six columns is
// thirty-two dispatches for every matrix: seventeen thousand of them and as
// many barriers, in one submission, which the driver's watchdog ends by
// resetting the card mid-pass. Sixty-four keeps a submission to something a
// card finishes.
//
// The cost is prefill throughput, and the fix is a tiled Golem product rather
// than a smaller number here.
const golemWidestPass = 64

// widestPass is the widest a submission of this pipeline may carry.
func (p *QwenPipeline) widestPass() int {
	if p.usesGolem() {
		return golemWidestPass
	}
	return qwenWide
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

	if b.golemEH != nil {
		p.prepare(r, b.preEH, 1)
		r.Barrier()
		p.golemProduct(r, b.golemEH, 1)
	} else {
		r.Dispatch(b.setEH, matvecGroups(s.Dim), unsafe.Pointer(&eh))
	}
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

// DeviceBytes is what this pipeline has allocated on the card so far: its own
// scratch, plus every block added to it. It is exact rather than estimated,
// because a window sized against an estimate of the scratch is what took the
// card down — the buffers a five-hundred-column pass needs are not small, and
// counting only the weights put the working set past the heap.
func (p *QwenPipeline) DeviceBytes() uint64 {
	var n uint64
	for _, b := range p.owned {
		if b != nil {
			n += b.size
		}
	}
	return n
}
