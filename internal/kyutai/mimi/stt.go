package mimi

// The Mimi that ships with Kyutai STT. It is the same codec family as
// pocket-tts's and not the same geometry: one SEANet stage more, four times the
// inner transformer, and a downsampling convolution of its own where the other
// folds sixteen steps inside the encoder. Only the encoding half is here — a
// transcriber never reconstructs sound.

import (
	"fmt"

	"github.com/ThiraSoft/golem/internal/kyutai/transformer"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// STTConfig is the codec that ships with the STT checkpoints.
var STTConfig = Config{
	LatentDim: 512,
	Channels:  512,
	Geometry: transformer.Geometry{
		DModel: 512, NumHeads: 8, DimFF: 2048, NumLayers: 8,
		Context: 250, LayerScale: true, MaxPeriod: 10000,
	},
	Ratios:     []int{4, 5, 6, 8},
	Downsample: 2,
	Prefix:     "",
}

type STTEncoder struct {
	Config Config

	input  nn.Conv1d
	stages []encoderStage
	output nn.Conv1d
	layers []*transformer.Layer
	down   nn.Conv1d
}

// sttStages names, for each stage, the module index of its residual block and
// of the strided convolution that follows, with the channel widths of both.
var sttStages = []struct{ block, shrink, in, out, kernel, stride int }{
	{1, 3, 64, 128, 8, 4},
	{4, 6, 128, 256, 10, 5},
	{7, 9, 256, 512, 12, 6},
	{10, 12, 512, 1024, 16, 8},
}

func LoadSTTEncoder(m *tensors.Model, cfg Config) (*STTEncoder, error) {
	e := &STTEncoder{Config: cfg}
	p := cfg.Prefix
	var err error
	if e.input, err = loadConv(m, p+"encoder.model.0.conv.conv", 1, 64, 7, 1, 1); err != nil {
		return nil, err
	}
	for _, s := range sttStages {
		var st encoderStage
		hidden := s.in / 2
		if st.block.conv1, err = loadConv(m, fmt.Sprintf("%sencoder.model.%d.block.1.conv.conv", p, s.block), s.in, hidden, 3, 1, 1); err != nil {
			return nil, err
		}
		if st.block.conv2, err = loadConv(m, fmt.Sprintf("%sencoder.model.%d.block.3.conv.conv", p, s.block), hidden, s.in, 1, 1, 1); err != nil {
			return nil, err
		}
		if st.shrink, err = loadConv(m, fmt.Sprintf("%sencoder.model.%d.conv.conv", p, s.shrink), s.in, s.out, s.kernel, s.stride, 1); err != nil {
			return nil, err
		}
		e.stages = append(e.stages, st)
	}
	if e.output, err = loadConv(m, p+"encoder.model.14.conv.conv", 1024, cfg.Channels, 3, 1, 1); err != nil {
		return nil, err
	}
	for i := 0; i < cfg.Geometry.NumLayers; i++ {
		l, err := transformer.LoadLayer(m, p+"encoder_transformer.transformer.", i, cfg.Geometry)
		if err != nil {
			return nil, err
		}
		e.layers = append(e.layers, l)
	}
	if e.down, err = loadConv(m, p+"downsample.conv.conv.conv", cfg.Channels, cfg.LatentDim, 2*cfg.Downsample, cfg.Downsample, 1); err != nil {
		return nil, err
	}
	return e, nil
}

// Latents runs the whole recording at once: no state, one matrix product per
// convolution. This is the path a file takes. The microphone's is Task 4's.
func (e *STTEncoder) Latents(samples []float32) ([]float32, int, error) {
	if len(samples)%SamplesPerFrame != 0 {
		return nil, 0, fmt.Errorf("mimi: %d samples is not a whole number of %d-sample frames", len(samples), SamplesPerFrame)
	}
	x, steps := e.input.Apply(samples, len(samples), e.input.NewState())
	for _, st := range e.stages {
		x = st.block.apply(x, steps, &blockState{s1: st.block.conv1.NewState(), s2: st.block.conv2.NewState()})
		nn.ELU(x)
		x, steps = st.shrink.Apply(x, steps, st.shrink.NewState())
	}
	nn.ELU(x)
	x, steps = e.output.Apply(x, steps, e.output.NewState())
	e.transformerSteps(x, steps)
	x, steps = e.down.Apply(x, steps, e.down.NewState())
	return x, steps, nil
}

// transformerSteps runs the inner transformer over every position, sharing one
// cache: the same arithmetic as one step at a time, with the indices written
// once. Copied in shape from Encoder.transformerSteps, which does the same for
// pocket-tts's codec.
func (e *STTEncoder) transformerSteps(x []float32, steps int) {
	g := e.Config.Geometry
	block := make([]float32, steps*g.DModel)
	for t := 0; t < steps; t++ {
		for c := 0; c < g.DModel; c++ {
			block[t*g.DModel+c] = x[c*steps+t]
		}
	}
	for _, layer := range e.layers {
		layer.Block(block, steps, transformer.NewCache(steps, layer.NumHeads, layer.HeadDim))
	}
	for t := 0; t < steps; t++ {
		for c := 0; c < g.DModel; c++ {
			x[c*steps+t] = block[t*g.DModel+c]
		}
	}
}

// STTState is the encoder's memory between chunks: one causal state per
// convolution, the block scratch, and the transformer's cache with the position
// it has reached. Everything the whole-recording path allocates and throws away.
type STTState struct {
	input  *nn.ConvState
	blocks []*blockState
	shrink []*nn.ConvState
	output *nn.ConvState
	caches []*transformer.Cache
	down   *nn.ConvState
}

func (e *STTEncoder) NewState() *STTState {
	g := e.Config.Geometry
	capacity := 16384
	if g.Context > capacity {
		capacity = g.Context
	}
	s := &STTState{
		input:  e.input.NewState(),
		output: e.output.NewState(),
		down:   e.down.NewState(),
		caches: make([]*transformer.Cache, len(e.layers)),
	}
	for i := range e.layers {
		s.caches[i] = transformer.NewCache(capacity, g.NumHeads, g.DModel/g.NumHeads)
	}
	for _, st := range e.stages {
		s.blocks = append(s.blocks, &blockState{s1: st.block.conv1.NewState(), s2: st.block.conv2.NewState()})
		s.shrink = append(s.shrink, st.shrink.NewState())
	}
	return s
}

// Push encodes one chunk. len(samples) must be a multiple of SamplesPerFrame.
// It returns the latents for the frames that chunk completed — possibly none —
// laid out channel by channel, and their count.
func (e *STTEncoder) Push(samples []float32, s *STTState) ([]float32, int) {
	if len(samples)%SamplesPerFrame != 0 {
		return nil, 0
	}
	x, steps := e.input.Apply(samples, len(samples), s.input)
	for i, st := range e.stages {
		x = st.block.apply(x, steps, s.blocks[i])
		nn.ELU(x)
		x, steps = st.shrink.Apply(x, steps, s.shrink[i])
	}
	nn.ELU(x)
	x, steps = e.output.Apply(x, steps, s.output)
	if steps > 0 {
		g := e.Config.Geometry
		block := make([]float32, steps*g.DModel)
		for t := 0; t < steps; t++ {
			for c := 0; c < g.DModel; c++ {
				block[t*g.DModel+c] = x[c*steps+t]
			}
		}
		for i, layer := range e.layers {
			layer.Block(block, steps, s.caches[i])
		}
		for t := 0; t < steps; t++ {
			for c := 0; c < g.DModel; c++ {
				x[c*steps+t] = block[t*g.DModel+c]
			}
		}
	}
	return e.down.Apply(x, steps, s.down)
}
