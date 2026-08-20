package mimi

// The Mimi encoder: sound in, latents out.
//
// It is the decoder read backwards. Where the decoder starts from one latent
// and multiplies the rate by four, then five, then six, the encoder divides it
// by four, then five, then six, and hands what is left to the same transformer
// geometry before folding sixteen steps into one latent.
//
// It does not stream, and that is deliberate. Encoding happens once per voice,
// on a recording that is already whole — there is no seam to preserve and no
// latency to hide. Handing each convolution the entire signal at once is also
// what makes it fast: a convolution given thousands of positions reaches
// nn.Conv1d's gathered form, which turns it into one matrix product per layer
// instead of a sweep per output channel.

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/pockettts/internal/transformer"
	"github.com/ThiraSoft/golem/tensors"
)

// Encoder holds everything needed to turn sound into latents.
type Encoder struct {
	Config Config

	input  nn.Conv1d // 1 -> 64, kernel 7
	stages []encoderStage
	output nn.Conv1d // 512 -> 512, kernel 3
	layers []*transformer.Layer
	down   nn.Conv1d // 512 -> 32, kernel 32, stride 16
}

// encoderStage is one reduction step: the residual block that shapes the signal,
// then the strided convolution that divides its rate.
type encoderStage struct {
	block  residualBlock
	shrink nn.Conv1d
}

// Encode turns samples in [-1, 1] at 24 kHz into one latent per frame, laid out
// channel by channel: latents[c*frames+f].
//
// The count of samples must be a multiple of SamplesPerFrame. The reference
// pads the recording up to one rather than encoding a partial frame, and so
// does the caller here — see pockettts.VoiceFromWAV.
func (e *Encoder) Encode(samples []float32) ([]float32, error) {
	if len(samples)%SamplesPerFrame != 0 {
		return nil, fmt.Errorf("%d samples is not a whole number of frames of %d",
			len(samples), SamplesPerFrame)
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("no samples to encode")
	}

	// Every convolution gets its own state, used for exactly one call: the
	// state is what carries the left padding, which is zero here, and nothing
	// follows this signal that could want its tail.
	x, steps := e.input.Apply(samples, len(samples), e.input.NewState())
	for _, st := range e.stages {
		x = st.block.apply(x, steps, &blockState{
			s1: st.block.conv1.NewState(), s2: st.block.conv2.NewState(),
		})
		nn.ELU(x)
		x, steps = st.shrink.Apply(x, steps, st.shrink.NewState())
	}
	nn.ELU(x)
	x, steps = e.output.Apply(x, steps, e.output.NewState())

	e.transformerSteps(x, steps)

	latents, frames := e.down.Apply(x, steps, e.down.NewState())
	if frames != len(samples)/SamplesPerFrame {
		return nil, fmt.Errorf("%d latents for %d frames of sound", frames, len(samples)/SamplesPerFrame)
	}
	return append([]float32(nil), latents...), nil
}

// transformerSteps runs the whole signal through the audio transformer, in
// place. Like the decoder's, it transposes: the convolutions lay out channel by
// channel, the transformer position by position.
func (e *Encoder) transformerSteps(x []float32, steps int) {
	D := e.Config.Geometry.DModel
	block := make([]float32, steps*D)
	for t := 0; t < steps; t++ {
		for c := 0; c < D; c++ {
			block[t*D+c] = x[c*steps+t]
		}
	}
	for _, layer := range e.layers {
		layer.Block(block, steps, transformer.NewCache(steps, layer.NumHeads, layer.HeadDim))
	}
	for t := 0; t < steps; t++ {
		for c := 0; c < D; c++ {
			x[c*steps+t] = block[t*D+c]
		}
	}
}

// LoadEncoder reads the encoder's weights.
func LoadEncoder(m *tensors.Model, cfg Config) (*Encoder, error) {
	e := &Encoder{Config: cfg}

	// The SEANet encoder is an indexed list, like the decoder, and the gaps in
	// the numbering are the activations, which carry no weights: 0 the input
	// convolution, then for each ratio a residual block and a strided
	// convolution, then the output convolution.
	const numFilters = 64
	var err error
	if e.input, err = loadConv(m, "mimi.encoder.model.0.conv", 1, numFilters, 7, 1, 1); err != nil {
		return nil, err
	}

	index := 1
	width := numFilters
	// The encoder runs the ratios in the opposite order to the decoder.
	for i := len(cfg.Ratios) - 1; i >= 0; i-- {
		ratio := cfg.Ratios[i]
		var st encoderStage
		block := fmt.Sprintf("mimi.encoder.model.%d.block.", index)
		if st.block.conv1, err = loadConv(m, block+"1.conv", width, width/2, 3, 1, 1); err != nil {
			return nil, err
		}
		if st.block.conv2, err = loadConv(m, block+"3.conv", width/2, width, 1, 1, 1); err != nil {
			return nil, err
		}
		// The stride divides the rate; the kernel is twice it, as in the
		// transposed convolution this mirrors.
		if st.shrink, err = loadConv(m, fmt.Sprintf("mimi.encoder.model.%d.conv", index+2),
			width, 2*width, 2*ratio, ratio, 1); err != nil {
			return nil, err
		}
		e.stages = append(e.stages, st)
		width *= 2
		index += 3
	}
	if e.output, err = loadConv(m, fmt.Sprintf("mimi.encoder.model.%d.conv", index+1),
		width, width, 3, 1, 1); err != nil {
		return nil, err
	}

	for i := 0; i < cfg.Geometry.NumLayers; i++ {
		c, err := transformer.LoadLayer(m, "mimi.encoder_transformer.transformer.", i, cfg.Geometry)
		if err != nil {
			return nil, fmt.Errorf("audio layer %d: %w", i, err)
		}
		e.layers = append(e.layers, c)
	}

	// The mirror of the decoder's grouped upsampling, except that it is not
	// grouped: it folds the 512 channels of sixteen steps into 32 numbers.
	if e.down, err = loadConv(m, "mimi.downsample.conv.conv",
		cfg.Channels, cfg.LatentDim, 2*StepsPerFrame, StepsPerFrame, 1); err != nil {
		return nil, err
	}
	// The only layer of the model that pads with its own edge rather than with
	// zeros, and the first latent of a voice depends on it.
	e.down.Replicate = true
	return e, nil
}
