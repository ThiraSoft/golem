package qwen

// The model: the embedding, twenty-eight blocks, the final norm, and a logit
// head that is the input embedding read the other way round.
//
// One token at a time, or a run of them. Generation is sequential by nature,
// and the prompt goes through the same path as everything else — a block sees a
// batch of positions, appends to a cache and reads it back up to itself.
//
// Gemma multiplies its embedding by the square root of the width on the way in;
// this model does not, which is a thing llama.cpp's graph says rather than a
// thing anyone should assume. There is no logit softcap and nothing suppressed.

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

type Model struct {
	Cfg *Config
	W   *Weights

	file        *tensors.GGUF
	cache       *Cache   // the active slot
	caches      []*Cache // every slot, when the caller asked for more than one
	slot        int
	slotContext int
	scratch     *Scratch
	embedded    map[int]*nn.Batch
	xs          [][]float32
	hidden      [][]float32
	outputs     [][]float32 // one per block, for the tests that locate a divergence
	batch       int
}

// Open maps a GGUF file and binds it. maxContext caps the cache; the file
// declares 40960, which no machine here would survive.
func Open(path string, maxContext int) (*Model, error) {
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		return nil, err
	}
	m, err := New(g, maxContext)
	if err != nil {
		g.Close()
		return nil, err
	}
	return m, nil
}

// New binds a GGUF the caller already opened, so that something reading the
// architecture first can hand the file straight over. The model takes
// ownership from there: its Close closes the file. A New that fails closes
// nothing, and the caller still holds the file it opened.
func New(g *tensors.GGUF, maxContext int) (*Model, error) {
	cfg, err := LoadConfig(g, maxContext)
	if err != nil {
		return nil, err
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		return nil, err
	}
	w.Repack()
	m := &Model{
		Cfg:      cfg,
		W:        w,
		file:     g,
		cache:    NewCache(cfg),
		scratch:  NewScratchFor(cfg, w),
		outputs:  make([][]float32, len(cfg.Blocks)),
		embedded: map[int]*nn.Batch{},
	}
	for i := range m.outputs {
		m.outputs[i] = make([]float32, cfg.Dim)
	}
	m.reserve(1)
	return m, nil
}

// reserve makes the model's own buffers, and the scratch behind them, wide
// enough for a batch of that size.
func (m *Model) reserve(batch int) {
	if batch <= m.batch {
		return
	}
	m.batch = batch
	m.scratch.Reserve(batch)
	m.xs = rows(batch, m.Cfg.Dim)
	m.hidden = rows(batch, m.Cfg.Dim)
}

func (m *Model) Close() error { return m.file.Close() }

// File is the mapped GGUF. The tokenizer lives in the same file as the weights,
// and a caller that wants both should not have to open it twice.
func (m *Model) File() *tensors.GGUF { return m.file }

// Reset forgets the conversation. The weights stay mapped.
func (m *Model) Reset() { m.cache.Reset() }

// Embed writes one token's embedding. No scale: llama.cpp's graph for this
// architecture records no inp_scaled, and Gemma's square root of the width
// belongs to Gemma.
func Embed(w *Weights, token int32, out []float32) {
	w.TokenEmbd.Row(int(token), out)
}

// Forward advances the model by one token and returns the hidden state after
// the final norm. The slice is reused: copy it if it has to outlive the next
// call.
func (m *Model) Forward(token int32, pos int) []float32 {
	return m.ForwardBatch([]int32{token}, pos)[0]
}

// ForwardBatch advances the model by a run of consecutive positions and returns
// one hidden state per token, after the final norm.
//
// The batch exists for one reason: a matrix of weights is read once for the
// whole of it. Generating an answer offers a batch of one and reads the weights
// per token; reading a prompt offers as many positions as it has, and reads the
// same weights for all of them. The arithmetic is identical either way — the
// results of a batch are what the same tokens would have given one at a time —
// and only the memory traffic changes.
func (m *Model) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	return m.ForwardMixed(tokens, Run(m.cache, startPos, len(tokens)))
}

// ForwardMixed is the same pass over a batch whose tokens need not belong to
// the same conversation: at says, for each of them, which cache it writes to
// and where. One read of the weights then serves several conversations, which
// is what lets a server answer more than one at a time.
func (m *Model) ForwardMixed(tokens []int32, at []Place) [][]float32 {
	cfg, w := m.Cfg, m.W
	batch := len(tokens)
	m.reserve(batch)

	embedded, ok := m.embedded[batch]
	if !ok {
		// Its own batch rather than one from the scratch space: the blocks
		// reuse the width-Dim activations for their norms, and the embedding
		// has to survive until it has been copied into the stream.
		embedded = nn.NewBatch(cfg.Dim, batch)
		embedded.BF16 = w.ActBF16
		m.embedded[batch] = embedded
	}
	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			Embed(w, tokens[t], embedded.F[t])
		}
	})

	xs := m.xs[:batch]
	for t := range tokens {
		copy(xs[t], embedded.F[t])
	}
	for i := range cfg.Blocks {
		bc := cfg.Blocks[i]
		ropes := m.scratch.RoPE(bc, at)
		Block(cfg, bc, &w.Blocks[i], ropes, at, m.scratch, xs)
		copy(m.outputs[i], xs[batch-1])
	}

	hidden := m.hidden[:batch]
	for t := range tokens {
		copy(hidden[t], xs[t])
		nn.RMSNormPlain(hidden[t], w.OutputNorm, cfg.Eps)
	}
	return hidden
}

// BlockOutput is what the given block last produced, for the tests that have to
// say which block a divergence began in.
func (m *Model) BlockOutput(block int) []float32 { return m.outputs[block] }

// Logits scores the whole vocabulary. The head is the input embedding read the
// other way round — this checkpoint ties them — so this reads the largest
// tensor in the file once per token.
func (m *Model) Logits(hidden []float32, out []float32) {
	if len(out) != m.Cfg.Vocab {
		panic(fmt.Sprintf("qwen: logits need %d entries, given %d", m.Cfg.Vocab, len(out)))
	}
	v := m.scratch.Batch(m.Cfg.Dim, 1)
	copy(v.F[0], hidden)
	// Rounds to bfloat16 when the head is bfloat16, and builds the Q8_0 form
	// otherwise.
	v.QuantizeColumnRange(0, 0, m.Cfg.Dim)
	// A K-quantized head wants the activation cut in superblocks rather than in
	// blocks of thirty-two.
	v.QuantizeK()
	m.W.TokenEmbd.MatVec(v, out)
}
