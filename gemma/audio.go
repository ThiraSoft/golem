package gemma

// The audio encoder.
//
// Two of them, behind one type. E2B's projector carries a twelve-block
// conformer; the 12B's carries a single projection and no encoder at all. What
// they share is their answer — one row per soft token, ProjDim wide — and the
// caller does not care which it got.

import "sync"

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
	s.transposed = make([]float32, (n+4)*dim)
	s.scores = make([]float32, cfg.Heads*blocks*cfg.Chunk*cfg.Context)
	s.rel = make([]float32, cfg.Heads*cfg.RPE*cfg.HeadDim)
	s.pos = make([]float32, cfg.RPE*dim)
	s.posTmp = make([]float32, cfg.RPE*dim)
	s.posOut = make([]float32, cfg.RPE*dim)
}
