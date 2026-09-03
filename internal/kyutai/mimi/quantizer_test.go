package mimi

import (
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
)

func TestQuantizerCodesAgainstReference(t *testing.T) {
	f := reference.Load(t, "stt")
	m := sttMimi(t)
	q, err := LoadQuantizer(m, STTConfig)
	if err != nil {
		t.Fatal(err)
	}
	latents := f.Read(t, "latents")
	frames := len(latents) / STTConfig.LatentDim
	want := f.Ints(t, "codes") // [32, frames], row-major
	latent := make([]float32, STTConfig.LatentDim)
	codes := make([]int, q.Codebooks)
	for fr := 0; fr < frames; fr++ {
		for c := range latent {
			latent[c] = latents[c*frames+fr]
		}
		q.Encode(latent, codes)
		for k, got := range codes {
			if got != want[k*frames+fr] {
				t.Fatalf("frame %d codebook %d: got %d, want %d", fr, k, got, want[k*frames+fr])
			}
		}
	}
}

func TestQuantizerLoadAndEncode(t *testing.T) {
	m := sttMimi(t)
	q, err := LoadQuantizer(m, STTConfig)
	if err != nil {
		t.Fatal(err)
	}
	if q.Codebooks != 32 {
		t.Fatalf("codebooks = %d, want 32", q.Codebooks)
	}
	latent := make([]float32, STTConfig.LatentDim)
	for i := range latent {
		latent[i] = float32(i%10) * 0.1
	}
	codes := make([]int, q.Codebooks)
	q.Encode(latent, codes)
	for k, code := range codes {
		if code < 0 || code >= 2048 {
			t.Fatalf("codebook %d produced out of range code %d", k, code)
		}
	}
}
