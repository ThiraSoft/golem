package gemma

// Giving a model ears, and putting a recording into a prompt.
//
// The same shape as the vision side, one step longer: a picture arrives as
// pixels and goes straight into the tower, where a recording arrives as a file
// some encoder wrote and has to be decoded, downmixed and resampled first. The
// model only ever sees sixteen kilohertz of mono, whatever was sent.

import (
	"bytes"
	"fmt"

	"github.com/ThiraSoft/golem/audio/decode"
	"github.com/ThiraSoft/golem/audio/mel"
	"github.com/ThiraSoft/golem/audio/resample"
)

// EncodeAudio decodes one sound file and runs the encoder over it. What comes
// back is one row per soft token, each as wide as the model's embedding.
func (m *Model) EncodeAudio(data []byte) ([][]float32, error) {
	if m.audio == nil {
		return nil, fmt.Errorf("gemma: this model was opened without an audio projector, so it cannot listen")
	}
	samples, rate, channels, err := decode.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("gemma: that sound file holds no samples")
	}
	mono := resample.Mono(samples, channels)
	return m.audio.Encode(resample.To(mono, rate, mel.Rate)), nil
}
