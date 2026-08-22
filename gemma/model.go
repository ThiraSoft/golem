package gemma

// The model: the embedding, thirty-five blocks, the final norm, and a logit
// head that is the input embedding read the other way round.
//
// One token at a time. Generation is sequential by nature, and the prompt goes
// through the same path as everything else — a block sees one position, appends
// to a cache and reads back what fits in its window.

import (
	"fmt"
	"math"

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
	ple         [][]float32
	outputs     [][]float32 // one per block, for the tests that locate a divergence
	batch       int
	vision      *VisionTower // nil until OpenVision
	visionFile  *tensors.GGUF
}

// Open maps a GGUF file and binds it. maxContext caps the cache; the file
// declares 131072, which no machine here would survive.
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
		scratch:  NewScratch(cfg),
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
	m.ple = rows(batch, len(m.Cfg.Blocks)*m.Cfg.PLEDim)
}

func (m *Model) Close() error {
	if m.visionFile != nil {
		m.visionFile.Close()
		m.visionFile = nil
	}
	return m.file.Close()
}

// File is the mapped GGUF. The tokenizer lives in the same file as the weights,
// and a caller that wants both should not have to open it twice.
func (m *Model) File() *tensors.GGUF { return m.file }

// Reset forgets the conversation. The weights stay mapped.
func (m *Model) Reset() { m.cache.Reset() }

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
// whole of it. Generating an answer offers a batch of one and reads a gigabyte
// per token; reading a prompt offers as many positions as it has, and reads the
// same gigabyte for all of them. The arithmetic is identical either way — the
// results of a batch are what the same tokens would have given one at a time,
// to the last bit — and only the memory traffic changes.
func (m *Model) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	return m.ForwardMixed(tokens, Run(m.cache, startPos, len(tokens)))
}

// ForwardMixed is the same pass over a batch whose tokens need not belong to
// the same conversation: at says, for each of them, which cache it writes to
// and where. One read of the weights then serves several conversations, which
// is what lets a server answer more than one at a time.
func (m *Model) ForwardMixed(tokens []int32, at []Place) [][]float32 {
	return m.ForwardEmbedded(tokens, nil, tokens, at)
}

// ForwardEmbedded is the same pass with the embedding of some positions given
// rather than looked up: embeds[t], when it is not nil, is the row that goes
// in at position t instead of the token's own. That is how a picture reaches
// the model — the vision tower's output rows go in at the placeholder
// positions.
//
// ple says which identifier each position contributes to the per-layer inputs,
// which is a lookup and needs one even where the embedding did not come from
// the table. llama.cpp answers that with the padding token for every position
// of an embedding batch, and passing zeros there is what agrees with it.
// Callers with no picture pass tokens for both.
func (m *Model) ForwardEmbedded(tokens []int32, embeds [][]float32, ple []int32, at []Place) [][]float32 {
	cfg, w := m.Cfg, m.W
	batch := len(tokens)
	m.reserve(batch)

	embedded, ok := m.embedded[batch]
	if !ok {
		// Its own batch rather than one from the scratch space: the blocks reuse
		// the width-Dim activations for their norms, and the embedding has to
		// survive until the per-layer inputs have been built from it.
		embedded = nn.NewBatch(cfg.Dim, batch)
		m.embedded[batch] = embedded
	}
	nn.InParallel(batch, batch*cfg.Dim*perPosition, func(first, last int) {
		for t := first; t < last; t++ {
			if embeds != nil && embeds[t] != nil {
				copy(embedded.F[t], embeds[t])
			} else {
				Embed(cfg, w, tokens[t], embedded.F[t])
			}
			embedded.QuantizeColumnRange(t, 0, cfg.Dim)
		}
	})
	perLayer := m.ple[:batch]
	PerLayerInputs(cfg, w, m.scratch, embedded, ple, perLayer)

	xs := m.xs[:batch]
	for t := range tokens {
		copy(xs[t], embedded.F[t])
	}
	blockPLE := make([][]float32, batch)
	for i := range cfg.Blocks {
		bc := cfg.Blocks[i]
		freqs := w.RoPEFreqs
		if bc.Window {
			freqs = nil // the frequency factors belong to the global blocks
		}
		if cfg.PLEDim > 0 {
			for t := range tokens {
				blockPLE[t] = perLayer[t][i*cfg.PLEDim : (i+1)*cfg.PLEDim]
			}
		}
		ropes := m.scratch.RoPE(bc, at, freqs)
		Block(cfg, bc, &w.Blocks[i], ropes, at, m.scratch, xs, blockPLE)
		copy(m.outputs[i], xs[batch-1])
	}

	hidden := m.hidden[:batch]
	for t := range tokens {
		copy(hidden[t], xs[t])
		nn.RMSNormPlain(hidden[t], w.OutputNorm, cfg.Eps)
	}
	return hidden
}

// BlockOutput is what the given block last produced. For the tests that have to
// say which block a divergence began in.
func (m *Model) BlockOutput(block int) []float32 { return m.outputs[block] }

// Logits scores the whole vocabulary. The head is the input embedding read the
// other way round — Gemma ties them — so this reads the largest tensor in the
// file once per token.
func (m *Model) Logits(hidden []float32, out []float32) {
	if len(out) != m.Cfg.Vocab {
		panic(fmt.Sprintf("gemma: logits need %d entries, given %d", m.Cfg.Vocab, len(out)))
	}
	v := m.scratch.Batch(m.Cfg.Dim, 1)
	copy(v.F[0], hidden)
	// The head is the only K-quantized product in the engine, and it wants the
	// activation cut in superblocks rather than in blocks of 32.
	v.QuantizeK()
	m.W.TokenEmbd.MatVec(v, out)
	nn.Softcap(out, m.Cfg.LogitSoftcap)
	m.suppress(out, nil)
}

// suppress forbids the tokens the file forbids, after the softcap, which is
// where llama.cpp adds its bias. ids says which token each entry of out scores;
// nil means out is the whole vocabulary in order.
func (m *Model) suppress(out []float32, ids []int32) {
	for _, id := range m.Cfg.Suppress {
		if ids == nil {
			if int(id) < len(out) {
				out[id] = float32(math.Inf(-1))
			}
			continue
		}
		for i, want := range ids {
			if want == id {
				out[i] = float32(math.Inf(-1))
			}
		}
	}
}

// LogitsAt scores the given tokens and nothing else, which is what a parity
// test wants: sixty-four rows read instead of a quarter of a million.
func (m *Model) LogitsAt(hidden []float32, ids []int32, out []float32) {
	row := make([]float32, m.Cfg.Dim)
	for i, id := range ids {
		m.W.TokenEmbd.Row(int(id), row)
		var sum float32
		for j, v := range row {
			sum += v * hidden[j]
		}
		out[i] = sum
	}
	nn.Softcap(out, m.Cfg.LogitSoftcap)
	m.suppress(out, ids)
}

// LogitsBatch scores several hidden states in one read of the head, which is
// what several conversations drawing at once need: the head is the largest
// matrix in the model and reading it once per conversation is what reading it
// once per token was before batches existed.
func (m *Model) LogitsBatch(hidden [][]float32, out [][]float32) {
	for _, o := range out {
		if len(o) != m.Cfg.Vocab {
			panic(fmt.Sprintf("gemma: logits need %d entries, given %d", m.Cfg.Vocab, len(o)))
		}
	}
	m.reserve(len(hidden))
	v := m.scratch.Batch(m.Cfg.Dim, len(hidden))
	for i := range hidden {
		copy(v.F[i], hidden[i])
	}
	v.QuantizeK()
	m.W.TokenEmbd.MatVecBatch(v, out)
	for _, o := range out {
		nn.Softcap(o, m.Cfg.LogitSoftcap)
		m.suppress(o, nil)
	}
}

// Argmax is the greedy choice. Ties go to the lower identifier, as llama.cpp
// does.
func Argmax(logits []float32) int32 {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return int32(best)
}
