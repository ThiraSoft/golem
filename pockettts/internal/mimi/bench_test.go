package mimi

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/pockettts/internal/reference"
	"github.com/ThiraSoft/golem/tensors"
)

func BenchmarkFrame(b *testing.B) {
	m, err := tensors.Open(weightsPath(b))
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	d, err := Load(m, DefaultConfig)
	if err != nil {
		b.Fatal(err)
	}
	L := DefaultConfig.LatentDim
	latent := make([]float32, L)
	for i := range latent {
		latent[i] = float32(i%7)*0.1 - 0.3
	}
	state := d.NewState(b.N + 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.Frame(latent, state)
	}
}

func weightsPath(b *testing.B) string {
	if p := os.Getenv("POCKET_TTS_WEIGHTS"); p != "" {
		return p
	}
	t, _ := filepath.Glob(filepath.Join(os.Getenv("HOME"),
		".cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/*/languages/french_24l/model.safetensors"))
	if len(t) == 0 {
		b.Skip("weights not found")
	}
	return t[0]
}

// BenchmarkEncode is the cloning path: a recording in, latents out.
//
// It is reported against the length of the recording, like the synthesis
// benchmark, because that is what a caller waits for — encoding twenty-eight
// seconds of speech is the whole of what preparing a voice costs.
func BenchmarkEncode(b *testing.B) {
	m, err := tensors.Open(reference.ModelPath(b, "french_24l"))
	if err != nil {
		b.Skip("no weights")
	}
	defer m.Close()
	e, err := LoadEncoder(m, DefaultConfig)
	if err != nil {
		b.Fatal(err)
	}

	// Ten seconds of a deterministic signal: long enough for every layer to be
	// dominated by its steady state rather than by its padding.
	const seconds = 10
	samples := make([]float32, seconds*24000)
	for i := range samples {
		samples[i] = float32(math.Sin(float64(i)*0.01)) * 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Encode(samples); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(seconds*b.N)/b.Elapsed().Seconds(), "x-real-time")
}
