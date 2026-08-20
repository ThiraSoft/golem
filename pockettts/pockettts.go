// Package pockettts synthesizes speech from text with the Kyutai Pocket TTS
// models, in pure Go.
//
// The engine loads the weights once — the file is memory-mapped, nothing is
// copied — then synthesizes as many times as asked. A voice is a precomputed
// state, produced by the reference Python daemon from a sound excerpt; the Go
// engine reads it back and therefore has no need for the audio encoder.
//
//	engine, err := pockettts.Open(pockettts.Options{
//		Weights:   ".../model.safetensors",
//		Tokenizer: ".../tokenizer.model",
//		Language:  "french_24l",
//	})
//	voice, err := engine.LoadVoice(".../voice.safetensors")
//	sound, err := engine.Synthesize("Bonjour le monde.", voice, nil)
package pockettts

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/ThiraSoft/golem/pockettts/internal/flowlm"
	"github.com/ThiraSoft/golem/pockettts/internal/mimi"
	"github.com/ThiraSoft/golem/pockettts/internal/text"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/sentencepiece"
)

// SampleRate is the sampling rate of the sound produced.
const SampleRate = 24000

// FramesPerSecond is the frame rate of the model.
const FramesPerSecond = 12.5

// Options describes what the engine needs in order to start.
type Options struct {
	Weights   string // model.safetensors of the language
	Tokenizer string // tokenizer.model
	// Language names which of the shipped models the weights belong to. It
	// decides the depth of the transformer and the text rules; getting it wrong
	// fails at load time rather than producing a wrong voice. Empty means
	// DefaultLanguage.
	Language string
}

// Engine holds the loaded weights. It is not safe for concurrent use:
// generation advances internal state. To synthesize in parallel, open one
// engine per goroutine — the weights being memory-mapped, the cost is that of
// the buffers, not that of the model.
type Engine struct {
	lang      Language
	model     *tensors.Model
	trans     *flowlm.Transformer
	flow      *flowlm.FlowNet
	decoder   *mimi.Decoder
	tokenizer *sentencepiece.Tokenizer
}

// Voice is a starting state: what the transformer had in mind after listening
// to the reference excerpt.
type Voice struct {
	state    *flowlm.State
	position int // where the voice stops, and where each segment resumes
}

// Open loads the weights and the tokenizer.
func Open(o Options) (*Engine, error) {
	lang, err := LookupLanguage(o.Language)
	if err != nil {
		return nil, err
	}
	m, err := tensors.Open(o.Weights)
	if err != nil {
		return nil, fmt.Errorf("weights: %w", err)
	}
	cfg := flowlm.ConfigFor(lang.Layers)

	trans, err := flowlm.Load(m, cfg)
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("transformer: %w", err)
	}
	flow, err := flowlm.LoadFlowNet(m, cfg)
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("flow net: %w", err)
	}
	dec, err := mimi.Load(m, mimi.DefaultConfig)
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("decoder: %w", err)
	}
	tok, err := sentencepiece.Load(o.Tokenizer)
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	return &Engine{lang: lang, model: m, trans: trans, flow: flow, decoder: dec, tokenizer: tok}, nil
}

// Close releases the memory mapping of the weights.
func (m *Engine) Close() error { return m.model.Close() }

// LoadVoice reads a precomputed voice state.
func (m *Engine) LoadVoice(path string) (*Voice, error) {
	// The capacity covers the voice and what a segment can add: at most fifty
	// text tokens, and the frames of some twenty seconds.
	state, err := m.trans.LoadVoice(path, 1024)
	if err != nil {
		return nil, err
	}
	return &Voice{state: state, position: state.Position()}, nil
}

// Language reports which language the engine was opened for.
func (m *Engine) Language() Language { return m.lang }

// rules are the text preparation rules of the engine's language.
func (m *Engine) rules() text.Rules {
	return text.Rules{
		DropSemicolons: m.lang.RemoveSemicolons,
		PadShortInputs: m.lang.PadShortInputs,
	}
}

// Settings tunes the generation. The zero value gives the model's own values,
// which are those of the reference daemon.
type Settings struct {
	Temperature    float64 // 0 -> 0.7: the variance of the starting noise
	EndThreshold   float64 // 0 -> -4: beyond it, the model declares itself done
	FramesAfterEnd int     // 0 -> 8: enough to let the sentence settle
	MaxTokens      int     // 0 -> 50: size of a segment
	Seed           uint64  // 0 -> random draw
	// Frame, if provided, receives each frame of samples as soon as it is
	// ready. That is what allows the sound to be played during generation.
	Frame func([]float32)

	// noise, if provided, replaces the random draw. Reserved for the
	// end-to-end test: it is the only way to compare against the reference a
	// generation that is non-reproducible by nature.
	noise func(frame int, into []float32)
}

func (r *Settings) defaults(lang Language) {
	if r.Temperature == 0 {
		r.Temperature = 0.7
	}
	if r.EndThreshold == 0 {
		r.EndThreshold = -4
	}
	if r.FramesAfterEnd == 0 {
		r.FramesAfterEnd = lang.FramesAfterEnd
	}
	if r.MaxTokens == 0 {
		r.MaxTokens = 50
	}
}

// Synthesize returns the samples of the text, in [-1, 1], at 24 kHz.
func (m *Engine) Synthesize(t string, voice *Voice, r *Settings) ([]float32, error) {
	if voice == nil {
		return nil, fmt.Errorf("no voice provided")
	}
	set := Settings{}
	if r != nil {
		set = *r
	}
	set.defaults(m.lang)

	seed := set.Seed
	if seed == 0 {
		seed = rand.Uint64()
	}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))

	segments := text.Segment(t, set.MaxTokens,
		func(s string) int { return len(m.tokenizer.Encode(s)) }, m.rules())
	if len(segments) == 0 {
		return nil, fmt.Errorf("empty text")
	}

	var sound []float32
	for _, segment := range segments {
		samples, err := m.synthesizeSegment(segment, voice, &set, rng)
		if err != nil {
			return nil, err
		}
		sound = append(sound, samples...)
	}
	return sound, nil
}

// synthesizeSegment handles a chunk that fits in one piece. It restarts from
// the untouched voice: the model was not trained to chain segments, and making
// them share a context degrades the diction.
func (m *Engine) synthesizeSegment(segment string, voice *Voice, r *Settings, rng *rand.Rand) ([]float32, error) {
	state := voice.state
	state.Rewind(voice.position)
	tokens := m.tokenizer.Encode(text.Prepare(segment, m.rules()))
	if len(tokens) == 0 {
		return nil, nil
	}
	m.trans.AdvancePrompt(tokens, state)

	// Without a limit, a model that never finds its end would generate
	// indefinitely. The bound is the daemon's: the estimated duration of the
	// text, plus two seconds.
	maxFrames := int((float64(len(tokens))/3.0 + 2.0) * FramesPerSecond)

	std := float32(math.Sqrt(r.Temperature))

	// The audio decoding runs on its own goroutine.
	//
	// Generating a latent and decoding it are two jobs that are independent from
	// one frame to the next: nothing prevents decoding frame n while the
	// transformer produces n+1. They even complement each other well — the
	// transformer mostly waits on memory, the decoder mostly on computation — so
	// that the cost of a frame drops from the sum of the two to the longer of
	// the two.
	latents := make(chan []float32, 4)
	chunks := make(chan []float32, 4)
	go func() {
		defer close(chunks)
		mimiState := m.decoder.NewState(maxFrames)
		for l := range latents {
			chunks <- m.decoder.Frame(l, mimiState)
		}
	}()

	// The collector assembles in order, without making the generator wait.
	done := make(chan []float32, 1)
	go func() {
		var sound []float32
		for chunk := range chunks {
			if r.Frame != nil {
				r.Frame(chunk)
			}
			sound = append(sound, chunk...)
		}
		done <- sound
	}()

	latent := append([]float32(nil), m.trans.BOSEmb...)
	noise := make([]float32, m.trans.Config.LatentDim)

	endFrame := -1
	for frame := 0; frame < maxFrames; frame++ {
		cond := m.trans.AdvanceLatent(latent, state)

		if float64(m.trans.EOSLogit(cond)) > r.EndThreshold && endFrame < 0 {
			endFrame = frame
		}
		if endFrame >= 0 && frame >= endFrame+r.FramesAfterEnd {
			break
		}

		if r.noise != nil {
			r.noise(frame, noise)
		} else {
			for i := range noise {
				noise[i] = float32(rng.NormFloat64()) * std
			}
		}
		next := make([]float32, m.trans.Config.LatentDim)
		m.flow.Latent(cond, noise, 1, next)

		latents <- next
		latent = next
	}
	close(latents)
	return <-done, nil
}
