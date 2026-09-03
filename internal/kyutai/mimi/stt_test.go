package mimi

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
	"github.com/ThiraSoft/golem/tensors"
)

// sttMimi opens the Mimi that ships with the STT checkpoint, or skips.
func sttMimi(t *testing.T) *tensors.Model {
	t.Helper()
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set: the STT checkpoint is not on this machine")
	}
	m, err := tensors.Open(dir + "/mimi-pytorch-e351c8d8@125.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestSTTEncoderAgainstReference(t *testing.T) {
	f := reference.Load(t, "stt")
	e, err := LoadSTTEncoder(sttMimi(t), STTConfig)
	if err != nil {
		t.Fatal(err)
	}
	latents, frames, err := e.Latents(f.Read(t, "audio"))
	if err != nil {
		t.Fatal(err)
	}
	if frames != 8 {
		t.Fatalf("frames = %d, want 8", frames)
	}
	reference.Compare(t, "latents", latents, f.Read(t, "latents"), 5e-5)
}

func TestSTTEncoderLoadAndRun(t *testing.T) {
	m := sttMimi(t)
	e, err := LoadSTTEncoder(m, STTConfig)
	if err != nil {
		t.Fatal(err)
	}
	// 8 frames of synthetic sine audio: 8 * 1920 = 15360 samples
	audio := make([]float32, 8*SamplesPerFrame)
	for i := range audio {
		audio[i] = float32(i%100) / 100.0 * 0.1
	}
	latents, frames, err := e.Latents(audio)
	if err != nil {
		t.Fatal(err)
	}
	if frames != 8 {
		t.Fatalf("frames = %d, want 8", frames)
	}
	if len(latents) != 8*STTConfig.LatentDim {
		t.Fatalf("latents length = %d, want %d", len(latents), 8*STTConfig.LatentDim)
	}
}
