package flowlm

// The flow_lm transformer: 24 causal layers, plus the input and output that
// frame them. It advances one position per call — that of a text token during
// the prompt fill, that of an audio frame afterwards.

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/pockettts/internal/transformer"
	"github.com/ThiraSoft/golem/tensors"
)

// Config describes the geometry of the flow_lm, read from the language's YAML.
type Config struct {
	transformer.Geometry
	LatentDim int // ldim: 32, the dimension of the audio latents
}

// ConfigFor is the geometry of a flow_lm of the given depth. Everything else is
// identical across the languages Kyutai ships: only the number of layers moves,
// 24 or 6.
func ConfigFor(layers int) Config {
	return Config{
		Geometry: transformer.Geometry{
			DModel: 1024, NumHeads: 16, DimFF: 4096, NumLayers: layers, MaxPeriod: 10000,
		},
		LatentDim: 32,
	}
}

// ConfigFrench24L is the geometry of french_24l, kept as the reference the
// parity tests are written against.
var ConfigFrench24L = ConfigFor(24)

// Transformer holds the layers and the input and output projections.
type Transformer struct {
	Config      Config
	Layers      []*transformer.Layer
	InputLinear nn.Linear // from the latent of 32 to d_model
	OutNorm     nn.LayerNorm
	OutEOS      nn.Linear
	EOSBias     float32
	BOSEmb      []float32 // starting latent, substituted for the first frame
	EmbMean     []float32 // denormalization statistics of the latents
	EmbStd      []float32
	Conditioner [][]float32 // table of text embeddings, indexed by token
	scratch     []float32
}

// State is the generation state: one KV cache per layer.
type State struct {
	Caches []*transformer.Cache
}

// NewState allocates an empty state for at most `capacity` positions.
func (t *Transformer) NewState(capacity int) *State {
	e := &State{Caches: make([]*transformer.Cache, len(t.Layers))}
	for i := range e.Caches {
		e.Caches[i] = transformer.NewCache(capacity, t.Config.NumHeads, t.Config.DModel/t.Config.NumHeads)
	}
	return e
}

// Position is the number of positions already consumed.
func (e *State) Position() int {
	if len(e.Caches) == 0 {
		return 0
	}
	return e.Caches[0].Position
}

// Advance passes one position through the 24 layers. x is modified in place.
func (t *Transformer) Advance(x []float32, state *State) {
	for i, c := range t.Layers {
		c.Step(x, state.Caches[i])
	}
}

// AdvanceText consumes one text embedding: same path, but the input comes from
// the conditioner's table rather than from a projected latent.
func (t *Transformer) AdvanceText(token int, state *State) []float32 {
	x := append([]float32(nil), t.Conditioner[token]...)
	t.Advance(x, state)
	return x
}

// AdvancePrompt consumes all the tokens of a segment in one go.
//
// Nothing requires feeding them one by one: they are all known in advance, and
// processing them together divides the re-reading of the 604 MB of weights by as
// much. That is the difference between an instant start and half a second of
// waiting before the first sound.
func (t *Transformer) AdvancePrompt(tokens []int, state *State) {
	if len(tokens) == 0 {
		return
	}
	D := t.Config.DModel
	block := make([]float32, len(tokens)*D)
	for i, tok := range tokens {
		copy(block[i*D:], t.Conditioner[tok])
	}
	for i, c := range t.Layers {
		c.Block(block, len(tokens), state.Caches[i])
	}
}

// AdvanceLatent consumes one audio latent: it is projected to d_model, crosses
// the layers, then goes through the output normalization. The returned vector is
// the conditioning of the flow net for the next frame.
func (t *Transformer) AdvanceLatent(latent []float32, state *State) []float32 {
	if t.scratch == nil {
		t.scratch = make([]float32, t.Config.DModel)
	}
	t.InputLinear.Apply(latent, t.scratch)
	x := append([]float32(nil), t.scratch...)
	t.Advance(x, state)
	t.OutNorm.Apply(x)
	return x
}

// EOSLogit says how finished the model judges the utterance to be.
func (t *Transformer) EOSLogit(cond []float32) float32 {
	var y [1]float32
	t.OutEOS.Apply(cond, y[:])
	return y[0] + t.EOSBias
}

// Load reads from the weights file everything the transformer uses.
func Load(m *tensors.Model, cfg Config) (*Transformer, error) {
	t := &Transformer{Config: cfg}

	for i := 0; i < cfg.NumLayers; i++ {
		c, err := transformer.LoadLayer(m, "flow_lm.transformer.", i, cfg.Geometry)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		t.Layers = append(t.Layers, c)
	}

	lin := func(name string, inputs, outputs int) (nn.Linear, error) {
		te, err := m.Get(name)
		if err != nil {
			return nn.Linear{}, err
		}
		if te.DType != "BF16" {
			return nn.Linear{}, fmt.Errorf("%s: dtype %s", name, te.DType)
		}
		return nn.Linear{Weights: te.Raw, Inputs: inputs, Outputs: outputs}, nil
	}
	f32 := func(name string) ([]float32, error) {
		te, err := m.Get(name)
		if err != nil {
			return nil, err
		}
		return te.F32()
	}

	var err error
	if t.InputLinear, err = lin("flow_lm.input_linear.weight", cfg.LatentDim, cfg.DModel); err != nil {
		return nil, err
	}
	if t.OutEOS, err = lin("flow_lm.out_eos.weight", cfg.DModel, 1); err != nil {
		return nil, err
	}
	bias, err := f32("flow_lm.out_eos.bias")
	if err != nil {
		return nil, err
	}
	t.EOSBias = bias[0]

	gain, err := f32("flow_lm.out_norm.weight")
	if err != nil {
		return nil, err
	}
	normBias, err := f32("flow_lm.out_norm.bias")
	if err != nil {
		return nil, err
	}
	t.OutNorm = nn.LayerNorm{Gain: gain, Bias: normBias, Eps: 1e-5}

	if t.BOSEmb, err = f32("flow_lm.bos_emb"); err != nil {
		return nil, err
	}
	if t.EmbMean, err = f32("flow_lm.emb_mean"); err != nil {
		return nil, err
	}
	if t.EmbStd, err = f32("flow_lm.emb_std"); err != nil {
		return nil, err
	}

	// The conditioner's table is small (4001 x 1024) and read on every token:
	// converting it once is better than reconverting it on every access.
	embed, err := m.Get("flow_lm.conditioner.embed.weight")
	if err != nil {
		return nil, err
	}
	flat, err := embed.F32()
	if err != nil {
		return nil, err
	}
	d := embed.Shape[1]
	t.Conditioner = make([][]float32, embed.Shape[0])
	for i := range t.Conditioner {
		t.Conditioner[i] = flat[i*d : (i+1)*d]
	}
	return t, nil
}

// Rewind brings the state back to an earlier position.
//
// Each segment must restart from the voice as it was, otherwise the second one
// would inherit the context of the first. Duplicating the state for that would
// cost four hundred megabytes per segment; rewinding it costs none, because the
// abandoned positions will be rewritten before being read again — attention
// never looks beyond the current position.
func (e *State) Rewind(position int) {
	for _, c := range e.Caches {
		c.Position = position
	}
}
