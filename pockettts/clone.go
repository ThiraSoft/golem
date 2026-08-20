package pockettts

// Cloning a voice: a recording in, a voice state out.
//
// The model imitates a voice by listening to it rather than by being trained on
// it. The recording is encoded by the Mimi encoder into one latent per frame,
// each latent is projected into the transformer's space, and the transformer
// reads them the way it would read the frames it had generated itself. What is
// left in its K/V caches is the voice — and that, not the sound, is what a
// voice file holds.

import (
	"fmt"
	"os"

	"github.com/ThiraSoft/golem/audio/wav"
	"github.com/ThiraSoft/golem/pockettts/internal/mimi"
)

// MaxCloneSeconds is the longest recording worth listening to.
//
// The model was not trained on longer ones: past about seventy seconds it emits
// its end almost immediately and produces a few seconds of sound instead of the
// whole text, without any error to say so. Twenty to thirty seconds is enough
// to carry a voice, and is what the reference tooling recommends.
const MaxCloneSeconds = 70

// VoiceFromWAV encodes a recording and returns the voice it holds.
//
// The file must be at the model's sampling rate. Resampling is not done here:
// it is a choice with audible consequences, and a caller who has to make it
// should make it knowingly rather than have it made in passing.
func (m *Engine) VoiceFromWAV(path string) (*Voice, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	samples, rate, err := wav.Read(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if rate != SampleRate {
		return nil, fmt.Errorf("%s is at %d Hz; the model listens at %d", path, rate, SampleRate)
	}
	return m.VoiceFromSamples(samples)
}

// VoiceFromSamples is VoiceFromWAV for a recording already in memory, in
// [-1, 1] at SampleRate.
func (m *Engine) VoiceFromSamples(samples []float32) (*Voice, error) {
	if !m.trans.CanClone() {
		return nil, fmt.Errorf("these weights cannot clone a voice: they are the build without voice cloning")
	}
	if len(samples) < mimi.SamplesPerFrame {
		return nil, fmt.Errorf("recording of %d samples: less than the %d of one frame",
			len(samples), mimi.SamplesPerFrame)
	}
	if max := MaxCloneSeconds * SampleRate; len(samples) > max {
		return nil, fmt.Errorf("recording of %.0f s: past %d s the model stops mid-sentence "+
			"without reporting anything; twenty to thirty seconds is what it wants",
			float64(len(samples))/SampleRate, MaxCloneSeconds)
	}

	if m.encoder == nil {
		enc, err := mimi.LoadEncoder(m.model, mimi.DefaultConfig)
		if err != nil {
			return nil, fmt.Errorf("encoder: %w", err)
		}
		m.encoder = enc
	}

	// The convolutions no longer accept a partial frame; the reference pads the
	// recording up to a whole one rather than dropping the tail, and silence is
	// what it pads with.
	if rest := len(samples) % mimi.SamplesPerFrame; rest != 0 {
		samples = append(append([]float32(nil), samples...),
			make([]float32, mimi.SamplesPerFrame-rest)...)
	}

	latents, err := m.encoder.Encode(samples)
	if err != nil {
		return nil, err
	}
	frames := len(samples) / mimi.SamplesPerFrame

	// The state holds the voice, and then the segments generated from it: the
	// same margin LoadVoice leaves.
	state := m.trans.NewState(frames + 1 + 1024)
	if err := m.trans.AdvanceVoice(latents, frames, state); err != nil {
		return nil, err
	}
	return &Voice{state: state, position: state.Position()}, nil
}

// SaveVoice writes a voice where LoadVoice can read it back, in the format the
// reference daemon wrote. Encoding a recording costs seconds; reading the file
// back costs milliseconds.
func (m *Engine) SaveVoice(path string, v *Voice) error {
	if v == nil {
		return fmt.Errorf("no voice to save")
	}
	return m.trans.SaveVoice(path, v.state)
}
