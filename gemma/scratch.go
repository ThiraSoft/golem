package gemma

// The buffers a forward pass reuses.
//
// A pass carries a batch of positions: one when an answer is being generated,
// as many as the prompt allows when one is being read. Every buffer here is
// therefore a row per position, allocated once for the widest batch the engine
// has been asked for and reused from then on — generation writes them a hundred
// thousand times, and nothing is allocated on that path.
//
// Vectors are kept by width because the model has several: 1536 for the
// residual stream, 2048 or 4096 for the concatenated heads, 6144 or 12288 for
// the feed forward.

import "github.com/ThiraSoft/golem/nn"

type Scratch struct {
	batch int

	batches map[[2]int]*nn.Batch        // by width and batch, a column per position
	ropes   map[float64][]*nn.RoPETable // by base, a table per position

	q      [][]float32 // Heads * HeadDim
	qh     [][]float32 // the query rounded to fp16, which is what ggml dots with
	k, v   [][]float32 // KVHeads * HeadDim
	scores [][]float32 // one row of visible positions per position and head

	// The embedding rounded to bfloat16, for the per-layer projection. Its own
	// buffer rather than one from the map: the vector it is rounded from is
	// itself of width Dim, and sharing would overwrite the caller's input.
	bf16 map[int]*nn.Batch // by batch size

	attn  [][]float32 // the attention output, Dim
	resid [][]float32 // the stream between the two halves, Dim
	ffn   [][]float32 // the feed-forward output, Dim
	pe    [][]float32 // the stream before the per-layer fold, Dim
	up    [][]float32 // the ungated half of the feed forward, at most maxFFN

	// The mixture's buffers. Route and ExpertFFN run once per block per
	// token, so nothing here is allocated on that path: the two batches are
	// per position, because an expert's matrix is read for one position and
	// no other, and a column taken from a wider batch would be strided.
	expIDs     [][]int32
	expWeights [][]float32
	expLogits  [][]float32
	// expGateUp holds both halves of one expert's first product: the gate in
	// the first ExpertFFN elements, the up in the next.
	expGateUp [][]float32
	expDown   [][]float32
	// The same two rows, each wrapped in the slice of one that MatVecRows
	// takes. The wrapper does not escape, so building it in the loop would
	// cost nothing; it is here because the loop reads better without it.
	expGateUpRow [][][]float32
	expDownRow   [][][]float32
	routerIn     map[int]*nn.Batch
	expertIn     []*nn.Batch
	expertMid    []*nn.Batch

	experts, expertsUsed, expertFFN int

	dim, maxContext               int
	maxQ, maxKV, maxFFN, maxHeads int
}

func NewScratch(cfg *Config) *Scratch {
	s := &Scratch{
		batches:    map[[2]int]*nn.Batch{},
		ropes:      map[float64][]*nn.RoPETable{},
		bf16:       map[int]*nn.Batch{},
		routerIn:   map[int]*nn.Batch{},
		dim:        cfg.Dim,
		maxContext: cfg.MaxContext,
	}
	for _, b := range cfg.Blocks {
		s.maxHeads = max(s.maxHeads, b.Heads)
		s.maxQ = max(s.maxQ, b.Heads*b.HeadDim)
		s.maxKV = max(s.maxKV, b.KVHeads*b.HeadDim)
		s.maxFFN = max(s.maxFFN, b.FFN)
	}
	s.experts, s.expertsUsed, s.expertFFN = cfg.Experts, cfg.ExpertsUsed, cfg.ExpertFFN
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
	s.resid = rows(batch, s.dim)
	s.ffn = rows(batch, s.dim)
	s.pe = rows(batch, s.dim)
	s.up = rows(batch, s.maxFFN)
	s.scores = rows(batch*s.maxHeads, s.maxContext)
	if s.experts > 0 {
		s.expIDs = ints(batch, s.expertsUsed)
		s.expWeights = rows(batch, s.expertsUsed)
		s.expLogits = rows(batch, s.experts)
		s.expGateUp = rows(batch, 2*s.expertFFN)
		s.expDown = rows(batch, s.dim)
		s.expGateUpRow = make([][][]float32, batch)
		s.expDownRow = make([][][]float32, batch)
		for t := 0; t < batch; t++ {
			s.expGateUpRow[t] = [][]float32{s.expGateUp[t]}
			s.expDownRow[t] = [][]float32{s.expDown[t]}
		}
		s.expertIn = batches(batch, s.dim)
		s.expertMid = batches(batch, s.expertFFN)
	}
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

func ints(count, width int) [][]int32 {
	r := make([][]int32, count)
	for i := range r {
		r[i] = make([]int32, width)
	}
	return r
}

// batches allocates one single-column batch per position.
func batches(count, width int) []*nn.Batch {
	b := make([]*nn.Batch, count)
	for i := range b {
		b[i] = nn.NewBatch(width, 1)
	}
	return b
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
		s.batches[key] = b
	}
	return b
}

// BF16 is the embedding rounded to bfloat16, for the per-layer projection.
func (s *Scratch) BF16(batch int) *nn.Batch {
	b, ok := s.bf16[batch]
	if !ok {
		b = nn.NewBatch(s.dim, batch)
		s.bf16[batch] = b
	}
	return b
}

// RoPE returns the rotation tables for this block's geometry, one per position
// of the batch and each made current for its own position. The model has two
// geometries, so the sines and cosines are computed twice per batch rather than
// once per head of every block.
func (s *Scratch) RoPE(bc BlockConfig, at []Place, freqs []float32) []*nn.RoPETable {
	t, ok := s.ropes[bc.RoPEBase]
	if !ok {
		t = tables(s.batch)
		s.ropes[bc.RoPEBase] = t
	}
	for i, p := range at {
		t[i].Prepare(bc.RoPEDims, p.Pos, bc.RoPEBase, freqs)
	}
	return t
}

// RouterIn is the batch the router's matrix reads: the residual, normalized
// without a gain and scaled. It is kept by batch size for the reason Batch is.
func (s *Scratch) RouterIn(batch int) *nn.Batch {
	b, ok := s.routerIn[batch]
	if !ok {
		b = nn.NewBatch(s.dim, batch)
		s.routerIn[batch] = b
	}
	return b
}

// ExpertIn and ExpertMid are one position's own single-column batches: the
// input an expert's gate and up read, and the gated half its down reads.
func (s *Scratch) ExpertIn(t int) *nn.Batch  { return s.expertIn[t] }
func (s *Scratch) ExpertMid(t int) *nn.Batch { return s.expertMid[t] }

// Q, K and V expose what the last call to Attention computed for the first
// position of its batch, for the tests that compare against llama.cpp's
// waypoints one step at a time.
func (s *Scratch) Q(b BlockConfig) []float32 { return s.q[0][:b.Heads*b.HeadDim] }
func (s *Scratch) K(b BlockConfig) []float32 { return s.k[0][:b.KVHeads*b.HeadDim] }
func (s *Scratch) V(b BlockConfig) []float32 { return s.v[0][:b.KVHeads*b.HeadDim] }

// Heads is the concatenation of the attention heads, before the output
// projection — llama.cpp calls this waypoint kqv_out.
func (s *Scratch) Heads(b BlockConfig) []float32 { return s.Batch(b.Heads*b.HeadDim, 1).F[0] }

// AttnOut is the residual stream after the attention half — llama.cpp's
// attn_out. PEIn is the stream just before the per-layer fold.
func (s *Scratch) AttnOut() []float32 { return s.resid[0] }
func (s *Scratch) PEIn() []float32    { return s.pe[0] }
