package qwen

// The buffers a forward pass reuses.
//
// A pass carries a batch of positions: one when an answer is being generated,
// as many as the prompt allows when one is being read. Every buffer here is
// therefore a row per position, allocated once for the widest batch the engine
// has been asked for and reused from then on — generation writes them a hundred
// thousand times, and nothing is allocated on that path.
//
// This model has one geometry rather than Gemma's two, so the maxima below all
// come out equal to their block's value. They are still computed rather than
// read off block zero: it costs a loop at startup and it is what makes the
// engine right for a checkpoint whose blocks differ.

import "github.com/ThiraSoft/golem/nn"

type Scratch struct {
	batch int

	batches map[[2]int]*nn.Batch        // by width and batch, a column per position
	ropes   map[float64][]*nn.RoPETable // by base, a table per position

	q      [][]float32 // Heads * HeadDim
	qh     [][]float32 // the query rounded to fp16, which is what ggml dots with
	k, v   [][]float32 // KVHeads * HeadDim
	scores [][]float32 // one row of visible positions per position and head

	attn [][]float32 // the attention output, Dim
	ffn  [][]float32 // the feed-forward output, Dim
	up   [][]float32 // the ungated half of the feed forward, at most maxFFN

	dim, maxContext               int
	maxQ, maxKV, maxFFN, maxHeads int

	// actBF16 is stamped onto every batch handed out, so that quantizing a
	// column also rounds it. See Weights.ActBF16.
	actBF16 bool
}

func NewScratch(cfg *Config) *Scratch { return newScratch(cfg, false) }

// NewScratchFor is NewScratch told what the weights want of its activations.
func NewScratchFor(cfg *Config, w *Weights) *Scratch { return newScratch(cfg, w.ActBF16) }

func newScratch(cfg *Config, actBF16 bool) *Scratch {
	s := &Scratch{
		batches:    map[[2]int]*nn.Batch{},
		ropes:      map[float64][]*nn.RoPETable{},
		dim:        cfg.Dim,
		maxContext: cfg.MaxContext,
		actBF16:    actBF16,
	}
	for _, b := range cfg.Blocks {
		s.maxHeads = max(s.maxHeads, b.Heads)
		s.maxQ = max(s.maxQ, b.Heads*b.HeadDim)
		s.maxKV = max(s.maxKV, b.KVHeads*b.HeadDim)
		s.maxFFN = max(s.maxFFN, b.FFN)
	}
	s.Reserve(1)
	return s
}

// Reserve makes every buffer wide enough for a batch of that size. It is called
// once per batch width the engine meets, which in practice is twice: one for
// generation, one for the prompt.
func (s *Scratch) Reserve(batch int) {
	if batch <= s.batch {
		return
	}
	s.batch = batch
	s.q = rows(batch, s.maxQ)
	s.qh = rows(batch, s.maxQ)
	s.k = rows(batch, s.maxKV)
	s.v = rows(batch, s.maxKV)
	s.attn = rows(batch, s.dim)
	s.ffn = rows(batch, s.dim)
	s.up = rows(batch, s.maxFFN)
	s.scores = rows(batch*s.maxHeads, s.maxContext)
	for base := range s.ropes {
		s.ropes[base] = tables(batch)
	}
}

func rows(count, width int) [][]float32 {
	r := make([][]float32, count)
	for i := range r {
		r[i] = make([]float32, width)
	}
	return r
}

func tables(count int) []*nn.RoPETable {
	t := make([]*nn.RoPETable, count)
	for i := range t {
		t[i] = &nn.RoPETable{}
	}
	return t
}

// Batch returns the reusable activations of the given width, one column per
// position, allocating on first use.
//
// There is one per batch size rather than one narrowed from the widest, because
// the interleaving of a batch is its stride: a lone column taken from a batch of
// thirty-two would have its blocks a kilobyte apart, and generation would walk
// them the long way round.
func (s *Scratch) Batch(width, batch int) *nn.Batch {
	key := [2]int{width, batch}
	b, ok := s.batches[key]
	if !ok {
		b = nn.NewBatch(width, batch)
		b.BF16 = s.actBF16
		s.batches[key] = b
	}
	return b
}

// RoPE returns the rotation tables for this block's geometry, one per position
// of the batch and each made current for its own position. Every block shares
// one base here, so the sines and cosines are computed once per batch rather
// than once per head of every block.
func (s *Scratch) RoPE(bc BlockConfig, at []Place) []*nn.RoPETable {
	t, ok := s.ropes[bc.RoPEBase]
	if !ok {
		t = tables(s.batch)
		s.ropes[bc.RoPEBase] = t
	}
	for i, p := range at {
		t[i].Prepare(bc.RoPEDims, p.Pos, bc.RoPEBase, nil)
	}
	return t
}

// Q, K and V expose what the last call to Attention computed for the first
// position of its batch, for the tests that compare against llama.cpp's
// waypoints one step at a time.
func (s *Scratch) Q(b BlockConfig) []float32 { return s.q[0][:b.Heads*b.HeadDim] }
func (s *Scratch) K(b BlockConfig) []float32 { return s.k[0][:b.KVHeads*b.HeadDim] }
func (s *Scratch) V(b BlockConfig) []float32 { return s.v[0][:b.KVHeads*b.HeadDim] }

// Heads is the concatenation of the attention heads, before the output
// projection — llama.cpp calls this waypoint kqv_out.
func (s *Scratch) Heads(b BlockConfig) []float32 { return s.Batch(b.Heads*b.HeadDim, 1).F[0] }
