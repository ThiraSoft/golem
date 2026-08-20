package mimi

import (
	"testing"

	"os"
	"path/filepath"

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
