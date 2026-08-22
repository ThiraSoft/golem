package gemma

// The audio encoder.
//
// Two of them, behind one type. E2B's projector carries a twelve-block
// conformer; the 12B's carries a single projection and no encoder at all. What
// they share is their answer — one row per soft token, ProjDim wide — and the
// caller does not care which it got.

import (
	"sync"

	"github.com/ThiraSoft/golem/audio/mel"
	"github.com/ThiraSoft/golem/nn"
)

// AudioTower is a projector's audio half, bound and ready to run.
type AudioTower struct {
	Cfg *AudioConfig
	W   *AudioWeights

	// scratches are the working buffers, one per encoding in flight. A server
	// asked for two recordings at once runs two, and two that shared a buffer
	// would write into each other's frames.
	scratches sync.Pool
}

// NewAudioTower binds a configuration to its weights.
func NewAudioTower(cfg *AudioConfig, w *AudioWeights) *AudioTower {
	return &AudioTower{Cfg: cfg, W: w}
}

// audioScratch is everything one pass over one chunk needs. It is sized for
// the positions it was made for and grown when a longer chunk arrives, which
// happens at most once: the first chunk of a recording is the longest one.
type audioScratch struct {
	n int

	norm, q, k, v []float32 // n by Dim
	ctx           []float32 // blocks*Chunk by Dim, the attention's answer
	branch        []float32 // n by Dim, what a branch hands back
	wide          []float32 // n by max(Dim, FFN), the projections' input
	ffn           []float32 // n by FFN
	gated         []float32 // n by 2*Dim, the convolution module's expansion
	gate          []float32 // n by Dim, what the gating leaves of it
	transposed    []float32 // Dim by n, the depthwise convolution's layout
	scores        []float32 // heads*blocks by Chunk*Context
	rel           []float32 // heads*RPE by HeadDim
	pos, posTmp   []float32 // RPE by Dim
	posOut        []float32 // RPE by Dim
}

// takeScratch borrows a scratch big enough for n positions.
func (a *AudioTower) takeScratch(n int) *audioScratch {
	s, _ := a.scratches.Get().(*audioScratch)
	if s == nil {
		s = &audioScratch{}
	}
	a.resize(s, n)
	return s
}

func (a *AudioTower) putScratch(s *audioScratch) { a.scratches.Put(s) }

func (a *AudioTower) resize(s *audioScratch, n int) {
	if s.n >= n {
		return
	}
	cfg := a.Cfg
	dim, ffn := cfg.Dim, cfg.FFN
	blocks := (n + cfg.Chunk - 1) / cfg.Chunk
	padded := blocks * cfg.Chunk

	s.n = n
	s.norm = make([]float32, n*dim)
	s.q = make([]float32, n*dim)
	s.k = make([]float32, n*dim)
	s.v = make([]float32, n*dim)
	s.ctx = make([]float32, padded*dim)
	s.branch = make([]float32, n*dim)
	s.wide = make([]float32, n*max(dim, ffn))
	s.ffn = make([]float32, n*ffn)
	s.gated = make([]float32, n*2*dim)
	s.gate = make([]float32, n*dim)
	s.gate = make([]float32, n*dim)
	s.transposed = make([]float32, (n+4)*dim)
	s.scores = make([]float32, cfg.Heads*blocks*cfg.Chunk*cfg.Context)
	s.rel = make([]float32, cfg.Heads*cfg.RPE*cfg.HeadDim)
	s.pos = make([]float32, cfg.RPE*dim)
	s.posTmp = make([]float32, cfg.RPE*dim)
	s.posOut = make([]float32, cfg.RPE*dim)
}

// Encode runs the whole front end and encoder over 16 kHz mono samples and
// returns one row per soft token, each Cfg.ProjDim wide.
//
// A recording longer than thirty seconds is cut into chunks and each is
// encoded on its own — the attention's geometry is local, so nothing is lost
// across a cut that was not already out of reach — and the rows are laid end
// to end.
func (a *AudioTower) Encode(samples []float32) [][]float32 {
	if a.Cfg.Unified {
		return a.encodeUnified(samples)
	}
	var rows [][]float32
	for _, chunk := range mel.Gemma4A(samples) {
		frames := len(chunk) / mel.Bins
		rows = append(rows, a.encodeChunk(chunk, frames)...)
	}
	return rows
}

// encodeChunk is one thirty-second chunk's mel through the whole encoder.
func (a *AudioTower) encodeChunk(melIn []float32, frames int) [][]float32 {
	cfg := a.Cfg
	x, n := a.preEncode(melIn, frames)
	if n == 0 {
		return nil
	}
	s := a.takeScratch(n)
	defer a.putScratch(s)

	for i := range a.W.Blocks {
		a.block(&a.W.Blocks[i], x, n, s)
	}

	// The tail: the output projection with its bias, an RMS norm that scales
	// by mm.a.soft_emb_norm when the file carries one — E2B's does not, and
	// the norm then stands alone — and the projection into the model's width.
	wide := make([]float32, n*a.W.OutProj.Outputs)
	a.W.OutProj.Apply(x[:n*cfg.Dim], s.wide, wide, n)
	width := a.W.OutProj.Outputs
	nn.InParallel(n, n*width, func(first, last int) {
		for p := first; p < last; p++ {
			row := wide[p*width : (p+1)*width]
			for i, v := range a.W.OutProjBias {
				row[i] += v
			}
			nn.RMSNormPlain(row, a.W.SoftEmbNorm, cfg.Eps)
		}
	})
	out := make([]float32, n*cfg.ProjDim)
	a.W.MMProj.Apply(wide, make([]float32, n*width), out, n)

	rows := make([][]float32, n)
	for p := range rows {
		rows[p] = out[p*cfg.ProjDim : (p+1)*cfg.ProjDim]
	}
	return rows
}

// block runs one conformer block in place.
//
// The order is the reference's: a half-step feed forward whose branch is
// halved before it is added, the attention, the convolution module, a second
// half-step feed forward, and the block's own norm. The two feed forwards have
// no gate — clip.cpp calls build_ffn with three null arguments, which is SiLU
// on the up projection alone and not the SwiGLU every other part of Gemma
// uses.
func (a *AudioTower) block(b *AudioBlock, x []float32, n int, s *audioScratch) {
	a.halfStepFFN(b.FFNNorm, b.FFNPostNorm, b.FFNUp, b.FFNDown, x, n, s)
	a.attention(b, x, s.branch, n, s)
	addInto(x[:n*a.Cfg.Dim], s.branch[:n*a.Cfg.Dim], 1)
	a.convModule(b, x, s.branch, n, s)
	addInto(x[:n*a.Cfg.Dim], s.branch[:n*a.Cfg.Dim], 1)
	a.halfStepFFN(b.FFNNorm1, b.FFNPostNorm1, b.FFNUp1, b.FFNDown1, x, n, s)
	dim := a.Cfg.Dim
	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			nn.RMSNormPlain(x[p*dim:(p+1)*dim], b.BlockNorm, a.Cfg.Eps)
		}
	})
}

// halfStepFFN adds half of a feed forward's answer back into the residual.
func (a *AudioTower) halfStepFFN(pre, post []float32, up, down VisionLinear, x []float32, n int, s *audioScratch) {
	cfg := a.Cfg
	dim, ffn := cfg.Dim, cfg.FFN
	norm := s.norm[:n*dim]
	copy(norm, x[:n*dim])
	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			nn.RMSNormPlain(norm[p*dim:(p+1)*dim], pre, cfg.Eps)
		}
	})
	hidden := s.ffn[:n*ffn]
	up.Apply(norm, s.wide, hidden, n)
	nn.InParallel(n*ffn, n*ffn*4, func(first, last int) {
		nn.SiLUGGMLRange(hidden, first, last)
	})
	branch := s.branch[:n*dim]
	down.Apply(hidden, s.wide, branch, n)
	nn.InParallel(n, n*dim, func(first, last int) {
		for p := first; p < last; p++ {
			nn.RMSNormPlain(branch[p*dim:(p+1)*dim], post, cfg.Eps)
		}
	})
	addInto(x[:n*dim], branch, 0.5)
}

// addInto adds a branch back into the residual, scaled.
func addInto(x, branch []float32, scale float32) {
	nn.InParallel(len(x), len(x), func(first, last int) {
		for i := first; i < last; i++ {
			x[i] += scale * branch[i]
		}
	})
}
