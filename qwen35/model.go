package qwen35

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/vk"
)

type Model struct {
	Cfg     *Config
	W       *Weights
	file    *tensors.GGUF
	cache   *Cache
	caches  []*Cache
	slot    int
	scratch *Scratch
	rope    *nn.RoPETable

	x      []float32
	hidden []float32

	// mtp is the prediction block's own scratch and cache, built the first
	// time a draft is asked for.
	mtp *MTPScratch

	// Vulkan GPU acceleration
	dev     *vk.Device
	gpuPipe *vk.QwenPipeline
	head    *vk.Q40Head
	headQ6K *vk.Q6KHead

	// vision is nil until a projector is opened. The tower runs once per
	// image and shares nothing with the text path.
	// imagePadID is the vocabulary's placeholder for one row of a picture,
	// which the engine is told rather than looks up: the vocabulary is opened
	// beside the model and not inside it.
	visionStartID int32
	imagePadID    int32
	visionEndID   int32

	// grids are the shapes the pictures encoded since the last prompt came
	// from, in that order. A row count does not say which way round a grid was.
	grids [][2]int

	vision *VisionTower
}

func Open(path string, maxContext int) (*Model, error) {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	return New(g, maxContext)
}

func New(g *tensors.GGUF, maxContext int) (*Model, error) {
	cfg, err := LoadConfig(g, maxContext)
	if err != nil {
		return nil, err
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		return nil, err
	}

	m := &Model{
		Cfg:     cfg,
		W:       w,
		file:    g,
		cache:   NewCache(cfg),
		scratch: NewScratch(cfg),
		rope:    &nn.RoPETable{},
		x:       make([]float32, cfg.Dim),
		hidden:  make([]float32, cfg.Dim),
	}
	return m, nil
}

func (m *Model) Close() error {
	// The tower first: what it holds was allocated on the device closeVulkan
	// is about to destroy, and a buffer freed after its device is a call into
	// a handle that no longer names anything.
	if m.vision != nil {
		m.vision.Close()
		m.vision = nil
	}
	m.closeVulkan()
	if m.file != nil {
		return m.file.Close()
	}
	return nil
}

func (m *Model) Reset() {
	m.cache.Reset()
	m.ResetMTP()
	if m.gpuPipe != nil {
		if err := m.gpuPipe.ResetState(); err != nil {
			panic(fmt.Sprintf("qwen35: cannot clear the GPU recurrent state: %v", err))
		}
	}
}

func (m *Model) Slots() int {
	if len(m.caches) == 0 {
		return 1
	}
	return len(m.caches)
}

func (m *Model) SlotContext() int {
	return m.Cfg.MaxContext
}

func (m *Model) SetSlots(n int) error {
	// The card holds one delta net's state per block, and it is not indexed by
	// slot. The attention blocks could be cut the way gemma's and qwen's now
	// are — the position buffer carries the slot and vk/attention.go lays one
	// ring a conversation — but a recurrent block has no ring to cut: its
	// state is a matrix a head that every token rewrites, and the copies the
	// speculation path makes of it would each need a slot too. Answering
	// several conversations off one state mixes them together, which is a
	// wrong answer and not a slow one, so it is refused rather than fallen
	// back from. Slots on the processor are unaffected.
	if n > 1 && m.gpuPipe != nil {
		return fmt.Errorf("qwen35: the GPU pipeline holds one conversation, not %d", n)
	}
	if n <= 1 {
		m.caches = nil
		m.slot = 0
		return nil
	}
	m.caches = make([]*Cache, n)
	for i := range m.caches {
		m.caches[i] = NewCache(m.Cfg)
	}
	m.cache = m.caches[0]
	m.slot = 0
	return nil
}

func (m *Model) UseSlot(i int) {
	m.slot = i
	if len(m.caches) > i {
		m.cache = m.caches[i]
	}
}

// X is the un-normed hidden state the last pass left, which is what the
// multi-token-prediction block reads.
func (m *Model) X() []float32 { return m.x }

// trunk is how many blocks the main pass runs: everything but the
// multi-token-prediction blocks at the end.
func (m *Model) trunk() int {
	n := len(m.Cfg.Blocks) - m.Cfg.NextN
	if n < 0 {
		return 0
	}
	return n
}

// Forward advances the model by one token at position pos and returns the
// hidden state under the model's output norm.
func (m *Model) Forward(token int32, pos int) []float32 {
	return m.ForwardBatch([]int32{token}, pos)[0]
}

// ForwardBatch processes a sequential run of tokens starting at startPos.
//
// On the card the run is carried up to eight tokens to a pass, because a pass
// reads every weight in the model once whatever it answers — a pass of two
// costs 1.056 of a pass of one. The delta net's recurrence still runs in order
// inside the pass; what is shared is the reading of the weights, which is the
// whole cost. WidthFor picks the widest binary that fits what is left, so a
// run ends on a pass of four or two rather than on a tail of single columns.
func (m *Model) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	out := make([][]float32, len(tokens))
	if m.gpuPipe != nil {
		// One row a column of the widest pass this run will actually take,
		// which is not the widest the pipeline can take: a card that reads a
		// prompt a hundred and twenty-eight positions at a time would
		// otherwise allocate and clear that many rows to draw one token.
		embeds := make([][]float32, m.gpuPipe.WidthFor(len(tokens)))
		for i := range embeds {
			embeds[i] = make([]float32, m.Cfg.Dim)
		}
		for t := 0; t < len(tokens); {
			n := m.gpuPipe.WidthFor(len(tokens) - t)
			at := make([]vk.QwenPlace, n)
			for c := 0; c < n; c++ {
				m.W.TokenEmbd.Row(int(tokens[t+c]), embeds[c])
				p := startPos + t + c
				at[c] = vk.QwenPlace{Pos: p, T: p, H: p, W: p}
			}
			hs, err := m.gpuPipe.ForwardPlaces(embeds[:n], at)
			if err != nil {
				panic(fmt.Sprintf("qwen35: the GPU pipeline failed at position %d: %v", startPos+t, err))
			}
			for c := 0; c < n; c++ {
				out[t+c] = append([]float32(nil), hs[c]...)
			}
			copy(m.x, m.gpuPipe.HiddenColumn(n-1))
			t += n
		}
		return out
	}
	places := Run(m.slot, startPos, len(tokens))
	for t, tok := range tokens {
		out[t] = m.step(tok, places[t])
	}
	return out
}

// ForwardSlots processes tokens across arbitrary slots and positions.
func (m *Model) ForwardSlots(tokens []int32, slots, positions []int) [][]float32 {
	at := make([]Place, len(tokens))
	for t := range tokens {
		p := positions[t]
		at[t] = Place{Slot: slots[t], Pos: p, T: p, H: p, W: p}
	}
	return m.ForwardPlaces(tokens, at)
}

// ForwardPlaces is ForwardSlots for a caller whose positions do not agree on
// every axis, which is what an image gives. Everything else arrives through
// Run, and gets places whose axes follow the cache index.
func (m *Model) ForwardPlaces(tokens []int32, at []Place) [][]float32 {
	out := make([][]float32, len(tokens))
	for t, tok := range tokens {
		m.UseSlot(at[t].Slot)
		out[t] = m.step(tok, at[t])
	}
	return out
}

// stepEmbedded is step for a position whose embedding is given rather than
// looked up, which is how a picture's rows reach the model.
func (m *Model) stepEmbedded(row []float32, at Place) []float32 {
	copy(m.x, row)

	if m.gpuPipe != nil {
		h, err := m.gpuPipe.ForwardPlaces([][]float32{m.x}, []vk.QwenPlace{at.gpu()})
		if err != nil {
			panic(fmt.Sprintf("qwen35: the GPU pipeline failed at position %d: %v", at.Pos, err))
		}
		copy(m.x, m.gpuPipe.Hidden())
		return append([]float32(nil), h[0]...)
	}

	for i, n := 0, m.trunk(); i < n; i++ {
		Block(m.Cfg, m.Cfg.Blocks[i], &m.W.Blocks[i], &m.cache.Blocks[i], m.rope, at, m.x, m.scratch)
	}
	h := make([]float32, m.Cfg.Dim)
	copy(h, m.x)
	nn.RMSNormPlain(h, m.W.OutputNorm, m.Cfg.Eps)
	return h
}

// step is one token through the trunk. The hidden state it returns is a fresh
// slice, because a caller holds on to it across the next token — the
// multi-token-prediction path holds two at once.
func (m *Model) step(token int32, at Place) []float32 {
	m.W.TokenEmbd.Row(int(token), m.x)

	if m.gpuPipe != nil {
		h, err := m.gpuPipe.ForwardPlaces([][]float32{m.x}, []vk.QwenPlace{at.gpu()})
		if err != nil {
			panic(fmt.Sprintf("qwen35: the GPU pipeline failed at position %d: %v", at.Pos, err))
		}
		// The un-normed state comes back too: it is what the MTP block reads,
		// and it is the residual the next token would continue from.
		copy(m.x, m.gpuPipe.Hidden())
		return append([]float32(nil), h[0]...)
	}

	for i, n := 0, m.trunk(); i < n; i++ {
		Block(m.Cfg, m.Cfg.Blocks[i], &m.W.Blocks[i], &m.cache.Blocks[i], m.rope, at, m.x, m.scratch)
	}

	h := make([]float32, m.Cfg.Dim)
	copy(h, m.x)
	nn.RMSNormPlain(h, m.W.OutputNorm, m.Cfg.Eps)
	return h
}

// Logits calculates the vocabulary logits from the final hidden state.
func (m *Model) Logits(hidden []float32, out []float32) {
	if len(out) != m.Cfg.Vocab {
		panic(fmt.Sprintf("qwen35: logits need %d entries, given %d", m.Cfg.Vocab, len(out)))
	}
	batchH := nn.NewBatch(m.Cfg.Dim, 1)
	copy(batchH.F[0], hidden)
	batchH.QuantizeK()

	if m.headQ6K != nil {
		if err := m.headQ6K.MatVec(batchH, 0, out); err == nil {
			return
		}
	}
	if m.head != nil {
		if err := m.head.MatVec(batchH, out); err == nil {
			return
		}
	}
	m.W.OutputHead.MatVec(batchH, out)
}

// LogitsBatch calculates logits for multiple hidden states in parallel.
func (m *Model) LogitsBatch(hidden [][]float32, out [][]float32) {
	batch := len(hidden)
	batchH := nn.NewBatch(m.Cfg.Dim, batch)
	for i := 0; i < batch; i++ {
		copy(batchH.F[i], hidden[i])
	}
	batchH.QuantizeK()

	for i := 0; i < batch; i++ {
		if m.headQ6K != nil {
			if err := m.headQ6K.MatVec(batchH, i, out[i]); err == nil {
				continue
			}
		}
		m.W.OutputHead.MatVec(batchH, out[i])
	}
}
