package flowlm

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

func BenchmarkAdvanceLatent(b *testing.B) {
	p := os.Getenv("POCKET_TTS_WEIGHTS")
	if p == "" {
		t, _ := filepath.Glob(filepath.Join(os.Getenv("HOME"),
			".cache/huggingface/hub/models--kyutai--pocket-tts/snapshots/*/languages/french_24l/model.safetensors"))
		if len(t) == 0 {
			b.Skip("weights not found")
		}
		p = t[0]
	}
	m, err := tensors.Open(p)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	tr, err := Load(m, ConfigFrench24L)
	if err != nil {
		b.Fatal(err)
	}
	state := tr.NewState(b.N + 8)
	latent := make([]float32, ConfigFrench24L.LatentDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tr.AdvanceLatent(latent, state)
	}
}
