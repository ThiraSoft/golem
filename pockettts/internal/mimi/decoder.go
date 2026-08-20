package mimi

// The Mimi decoder: from latent to sound.
//
// Every frame brings a latent of 32 values and must yield 1920 samples at
// 24 kHz — one eightieth of a second. The path climbs in stages: a projection
// brings the latent up to 512 channels, a transposed convolution spreads it over
// sixteen internal steps, a small transformer puts them in context, then the
// SEANet decoder climbs back to the audio rate through three successive
// expansions, x6, x5 and x4.
//
// Everything is in streaming mode: each convolution keeps the context the next
// frame will ask for. That is what allows the sound to be produced as generation
// goes rather than at the end, and it is the point where an error is heard
// rather than seen — a break at the seam between frames.

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/pockettts/internal/transformer"
	"github.com/ThiraSoft/golem/tensors"
)

// StepsPerFrame is the number of internal steps a latent produces before the
// SEANet decoder: the ratio between the rate of the audio transformer (200 Hz)
// and that of the frames (12.5 Hz).
const StepsPerFrame = 16

// SamplesPerFrame is what the decoder returns for one latent.
const SamplesPerFrame = 1920

// Config describes the geometry of the decoder.
type Config struct {
	LatentDim int // 32
	Channels  int // 512, the width of the decoder
	Geometry  transformer.Geometry
	Ratios    []int // 6, 5, 4: the successive expansion factors
}

// DefaultConfig is the geometry of the decoder. It carries no language name
// because the codec is the same for all of them: every config Kyutai ships
// describes this exact decoder.
var DefaultConfig = Config{
	LatentDim: 32,
	Channels:  512,
	Geometry: transformer.Geometry{
		DModel: 512, NumHeads: 8, DimFF: 2048, NumLayers: 2,
		Context: 250, LayerScale: true, MaxPeriod: 10000,
	},
	Ratios: []int{6, 5, 4},
}

// residualBlock is a SEANet block: two convolutions around an activation, added
// to their input.
type residualBlock struct {
	conv1, conv2 nn.Conv1d
}

// blockState holds the context of the block's two convolutions, and the copy
// the block works on: the input has to survive until the addition at the end,
// so the activation cannot be applied in place — but the copy itself is the
// same size on every frame and belongs in the state like the rest.
type blockState struct {
	s1, s2  *nn.ConvState
	scratch []float32
}

func (b residualBlock) apply(x []float32, steps int, s *blockState) []float32 {
	if cap(s.scratch) < len(x) {
		s.scratch = make([]float32, len(x))
	}
	v := s.scratch[:len(x)]
	copy(v, x)
	nn.ELU(v)
	v, n := b.conv1.Apply(v, steps, s.s1)
	nn.ELU(v)
	v, n = b.conv2.Apply(v, n, s.s2)
	if n != steps {
		panic(fmt.Sprintf("residual block: %d steps out for %d in", n, steps))
	}
	for i := range v {
		v[i] += x[i]
	}
	return v
}

// stage is one expansion step: the transposed convolution that multiplies the
// rate, then the residual block that shapes it.
type stage struct {
	expand nn.ConvTranspose1d
	block  residualBlock
}

// Decoder holds everything needed to turn latents into sound.
type Decoder struct {
	Config Config

	projection nn.Conv1d // 32 -> 512, kernel 1: the "dequantization"
	upsample   nn.ConvTranspose1d
	layers     []*transformer.Layer
	input      nn.Conv1d // 512 -> 512, kernel 7
	stages     []stage
	output     nn.Conv1d // 64 -> 1, kernel 3
	EmbMean    []float32 // denormalization statistics of the latents
	EmbStd     []float32
}

// State holds the streaming context of every layer.
type State struct {
	projection *nn.ConvState
	upsample   *nn.ConvTState
	caches     []*transformer.Cache
	input      *nn.ConvState
	stages     []struct {
		expand *nn.ConvTState
		block  blockState
	}
	output *nn.ConvState

	// The two buffers Frame itself needs: the denormalized latent, and the
	// transposed block the audio transformer works in.
	denorm []float32
	block  []float32
}

// NewState prepares a decoding run, for at most `frames` frames.
func (d *Decoder) NewState(frames int) *State {
	e := &State{
		projection: d.projection.NewState(),
		upsample:   d.upsample.NewState(),
		input:      d.input.NewState(),
		output:     d.output.NewState(),
	}
	for _, c := range d.layers {
		e.caches = append(e.caches, transformer.NewCache(
			frames*StepsPerFrame+StepsPerFrame, c.NumHeads, c.HeadDim))
	}
	e.stages = make([]struct {
		expand *nn.ConvTState
		block  blockState
	}, len(d.stages))
	for i, st := range d.stages {
		e.stages[i].expand = st.expand.NewState()
		e.stages[i].block.s1 = st.block.conv1.NewState()
		e.stages[i].block.s2 = st.block.conv2.NewState()
	}
	return e
}

// Frame decodes one latent and returns its samples, in [-1, 1].
func (d *Decoder) Frame(latent []float32, state *State) []float32 {
	// The flow_lm works on normalized latents; the codec expects the original
	// magnitudes.
	if cap(state.denorm) < len(latent) {
		state.denorm = make([]float32, len(latent))
	}
	denorm := state.denorm[:len(latent)]
	for i, v := range latent {
		denorm[i] = v*d.EmbStd[i] + d.EmbMean[i]
	}

	x, steps := d.projection.Apply(denorm, 1, state.projection)
	x, steps = d.upsample.Apply(x, steps, state.upsample)

	// The audio transformer works position by position; it therefore has to be
	// transposed, since the convolutions lay things out channel by channel.
	d.transformerSteps(x, steps, state)

	x, steps = d.input.Apply(x, steps, state.input)
	for i, st := range d.stages {
		nn.ELU(x)
		x, steps = st.expand.Apply(x, steps, state.stages[i].expand)
		x = st.block.apply(x, steps, &state.stages[i].block)
	}
	nn.ELU(x)
	x, steps = d.output.Apply(x, steps, state.output)

	// The one buffer that is not reused. The caller decodes on its own
	// goroutine and keeps several frames in flight, so the samples have to
	// belong to it; two thousand of them are nothing beside the six megabytes
	// a frame used to allocate.
	return append([]float32(nil), x...)
}

// transformerSteps passes the sixteen internal steps through the audio
// transformer, in place.
//
// The sixteen positions are processed together, not one after another: one
// position at a time, each layer would re-read its six million weights sixteen
// times per frame — more than the whole flow_lm, for a tenth of the computation.
func (d *Decoder) transformerSteps(x []float32, steps int, state *State) {
	D := d.Config.Geometry.DModel
	// The convolutions lay out by channel, the transformer by position.
	if cap(state.block) < steps*D {
		state.block = make([]float32, steps*D)
	}
	block := state.block[:steps*D]
	for t := 0; t < steps; t++ {
		for c := 0; c < D; c++ {
			block[t*D+c] = x[c*steps+t]
		}
	}
	for i, layer := range d.layers {
		layer.Block(block, steps, state.caches[i])
	}
	for t := 0; t < steps; t++ {
		for c := 0; c < D; c++ {
			x[c*steps+t] = block[t*D+c]
		}
	}
}

// Load reads the decoder's weights.
func Load(m *tensors.Model, cfg Config) (*Decoder, error) {
	ld := transformer.Loader{M: m}
	d := &Decoder{Config: cfg}

	var err error
	if d.projection, err = loadConv(m, "mimi.quantizer.output_proj", cfg.LatentDim, cfg.Channels, 1, 1, 1); err != nil {
		return nil, err
	}
	// Grouped upsampling: each channel is expanded independently.
	if d.upsample, err = loadConvT(m, "mimi.upsample.convtr.convtr", cfg.Channels, cfg.Channels, 2*StepsPerFrame, StepsPerFrame, cfg.Channels); err != nil {
		return nil, err
	}
	// One latent at a time reaches the upsampling; StepsPerFrame reach the
	// first expansion, and each ratio multiplies what reaches the next.
	d.upsample.Prepare(1)

	for i := 0; i < cfg.Geometry.NumLayers; i++ {
		c, err := transformer.LoadLayer(m, "mimi.decoder_transformer.transformer.", i, cfg.Geometry)
		if err != nil {
			return nil, fmt.Errorf("audio layer %d: %w", i, err)
		}
		d.layers = append(d.layers, c)
	}

	// The SEANet decoder is an indexed list: input conv, then for each ratio a
	// transposed convolution and a residual block, then the output convolution.
	const numFilters = 64
	channels := cfg.Channels
	if d.input, err = loadConv(m, "mimi.decoder.model.0.conv", channels, channels, 7, 1, 1); err != nil {
		return nil, err
	}
	index := 2
	width := numFilters * (1 << len(cfg.Ratios)) // 512 at the input of the first stage
	steps := StepsPerFrame                       // what reaches the first expansion
	for _, ratio := range cfg.Ratios {
		var st stage
		if st.expand, err = loadConvT(m, fmt.Sprintf("mimi.decoder.model.%d.convtr", index),
			width, width/2, 2*ratio, ratio, 1); err != nil {
			return nil, err
		}
		st.expand.Prepare(steps)
		steps *= ratio
		width /= 2
		block := fmt.Sprintf("mimi.decoder.model.%d.block.", index+1)
		if st.block.conv1, err = loadConv(m, block+"1.conv", width, width/2, 3, 1, 1); err != nil {
			return nil, err
		}
		if st.block.conv2, err = loadConv(m, block+"3.conv", width/2, width, 1, 1, 1); err != nil {
			return nil, err
		}
		d.stages = append(d.stages, st)
		index += 3
	}
	if d.output, err = loadConv(m, fmt.Sprintf("mimi.decoder.model.%d.conv", index), numFilters, 1, 3, 1, 1); err != nil {
		return nil, err
	}

	ld.Prefix = "flow_lm."
	if d.EmbMean, err = ld.Vector("emb_mean"); err != nil {
		return nil, err
	}
	if d.EmbStd, err = ld.Vector("emb_std"); err != nil {
		return nil, err
	}
	return d, nil
}

func loadConv(m *tensors.Model, name string, inputs, outputs, kernel, stride, dilation int) (nn.Conv1d, error) {
	t, err := m.Get(name + ".weight")
	if err != nil {
		return nn.Conv1d{}, err
	}
	want := []int{outputs, inputs, kernel}
	if len(t.Shape) != 3 || t.Shape[0] != want[0] || t.Shape[1] != want[1] || t.Shape[2] != want[2] {
		return nn.Conv1d{}, fmt.Errorf("%s: shape %v, want %v", name, t.Shape, want)
	}
	weights, err := t.F32()
	if err != nil {
		return nn.Conv1d{}, err
	}
	c := nn.Conv1d{Weights: weights, Inputs: inputs, Outputs: outputs, Kernel: kernel, Stride: stride, Dilation: dilation}
	if b, err := m.Get(name + ".bias"); err == nil {
		if c.Bias, err = b.F32(); err != nil {
			return nn.Conv1d{}, err
		}
	}
	return c, nil
}

func loadConvT(m *tensors.Model, name string, inputs, outputs, kernel, stride, groups int) (nn.ConvTranspose1d, error) {
	t, err := m.Get(name + ".weight")
	if err != nil {
		return nn.ConvTranspose1d{}, err
	}
	want := []int{inputs, outputs / groups, kernel}
	if len(t.Shape) != 3 || t.Shape[0] != want[0] || t.Shape[1] != want[1] || t.Shape[2] != want[2] {
		return nn.ConvTranspose1d{}, fmt.Errorf("%s: shape %v, want %v", name, t.Shape, want)
	}
	weights, err := t.F32()
	if err != nil {
		return nn.ConvTranspose1d{}, err
	}
	c := nn.ConvTranspose1d{Weights: weights, Inputs: inputs, Outputs: outputs, Kernel: kernel, Stride: stride, Groups: groups}
	if b, err := m.Get(name + ".bias"); err == nil {
		if c.Bias, err = b.F32(); err != nil {
			return nn.ConvTranspose1d{}, err
		}
	}
	return c, nil
}
