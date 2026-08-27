package qwen35

import "github.com/ThiraSoft/golem/nn"

// Scratch holds every buffer a block needs for one token. Nothing here is
// allocated per token: a 27B model runs sixty-five blocks per token, and a
// fresh 17408-float slice in each of them is more time in the allocator than in
// the arithmetic.
type Scratch struct {
	batchX    *nn.Batch // the block input, quantized for the projections
	batchAttn *nn.Batch // the attention output, feeding the O projection
	batchYSSM *nn.Batch // the delta-net output, feeding the SSM out projection
	batchFFN  *nn.Batch // the SwiGLU activation, feeding the down projection

	normed []float32 // the pre-norm of the block input
	subOut []float32 // what the mixer returns, before the residual

	// SSM scratch
	qkv     []float32 // [10240]
	gate    []float32 // [6144]
	convOut []float32 // [10240]
	alpha   []float32 // [48]
	beta    []float32 // [48]
	decay   []float32 // [48]
	ySSM    []float32 // [6144]

	// Full attention scratch
	qFull  []float32 // [24*256*2 = 12288], per head [q | gate]
	k      []float32 // [4*256 = 1024]
	v      []float32 // [4*256 = 1024]
	o      []float32 // [6144]
	scores []float32 // [MaxContext]

	// FFN scratch
	ffnUp   []float32
	ffnDown []float32
}

func NewScratch(cfg *Config) *Scratch {
	convDim, inner, heads, headDim, kvHeads, ffn, rank := 0, 0, 0, 0, 0, 0, 0
	for _, bc := range cfg.Blocks {
		convDim = max(convDim, bc.SSMStateSize*bc.SSMGroupCount*2+bc.SSMInnerSize)
		inner = max(inner, bc.SSMInnerSize)
		rank = max(rank, bc.SSMTimeStepRank)
		heads = max(heads, bc.Heads)
		headDim = max(headDim, bc.HeadDim)
		kvHeads = max(kvHeads, bc.KVHeads)
		ffn = max(ffn, bc.FFN)
	}
	qDim := heads * headDim
	mixer := max(inner, qDim)

	return &Scratch{
		batchX:    nn.NewBatch(cfg.Dim, 1),
		batchAttn: nn.NewBatch(qDim, 1),
		batchYSSM: nn.NewBatch(inner, 1),
		batchFFN:  nn.NewBatch(ffn, 1),

		normed: make([]float32, cfg.Dim),
		subOut: make([]float32, cfg.Dim),

		qkv:     make([]float32, convDim),
		gate:    make([]float32, inner),
		convOut: make([]float32, convDim),
		alpha:   make([]float32, rank),
		beta:    make([]float32, rank),
		decay:   make([]float32, rank),
		ySSM:    make([]float32, inner),

		qFull:  make([]float32, qDim*2),
		k:      make([]float32, kvHeads*headDim),
		v:      make([]float32, kvHeads*headDim),
		o:      make([]float32, mixer),
		scores: make([]float32, cfg.MaxContext),

		ffnUp:   make([]float32, ffn),
		ffnDown: make([]float32, cfg.Dim),
	}
}

// SetInput loads the block input and quantizes it once for every projection
// that reads it.
func (s *Scratch) SetInput(x []float32) {
	copy(s.batchX.F[0], x)
	quantize(s.batchX)
}

// load points a batch at a vector the caller already filled and refreshes every
// quantized form the batch carries. The vector is not copied: it is the
// scratch buffer the mixer just wrote, and the products read it in place.
//
// Refreshing matters more than it looks. A product against Q4_0 weights reads
// the Q8_0 form and never looks at the floats, so a batch whose floats were
// replaced without a requantization contributes whatever the previous token
// left behind — or zeros, on the first one.
func load(b *nn.Batch, v []float32) {
	b.F[0] = v
	quantize(b)
}

func quantize(b *nn.Batch) {
	b.QuantizeColumnRange(0, 0, b.Width)
	if b.QK != nil {
		b.QuantizeK()
	}
}
