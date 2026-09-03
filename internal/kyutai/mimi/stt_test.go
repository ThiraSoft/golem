package mimi

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
	"github.com/ThiraSoft/golem/nn"
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
	audio := f.Read(t, "audio")
	x, steps := e.input.Apply(audio, len(audio), e.input.NewState())
	for _, st := range e.stages {
		x = st.block.apply(x, steps, &blockState{s1: st.block.conv1.NewState(), s2: st.block.conv2.NewState()})
		nn.ELU(x)
		x, steps = st.shrink.Apply(x, steps, st.shrink.NewState())
	}
	nn.ELU(x)
	x, steps = e.output.Apply(x, steps, e.output.NewState())
	reference.Compare(t, "encoder", x, f.Read(t, "encoder"), 5e-5)

	e.transformerSteps(x, steps)
	reference.Compare(t, "encoder_transformer", x, f.Read(t, "encoder_transformer"), 2.5e-1)

	latents, frames := e.down.Apply(x, steps, e.down.NewState())
	if frames != 8 {
		t.Fatalf("frames = %d, want 8", frames)
	}
	reference.Compare(t, "latents", latents, f.Read(t, "latents"), 5e-2)
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

// TestStreamingMatchesWhole is the invariant the whole live path rests on: a
// recording pushed frame by frame must give the same latents as the same
// recording handed over at once. A convolution whose causal state remembers one
// sample too few does not crash — it degrades a transcript in a way that reads
// as the model being mediocre. This is what catches it.
func TestStreamingMatchesWhole(t *testing.T) {
	m := sttMimi(t)
	e, err := LoadSTTEncoder(m, STTConfig)
	if err != nil {
		t.Fatal(err)
	}
	framesCount := 8
	audio := make([]float32, framesCount*SamplesPerFrame)
	for i := range audio {
		audio[i] = float32(i%100) / 100.0 * 0.1
	}
	whole, frames, err := e.Latents(audio)
	if err != nil {
		t.Fatal(err)
	}

	state := e.NewState()
	var got []float32
	for i := 0; i < len(audio); i += SamplesPerFrame {
		out, n := e.Push(audio[i:i+SamplesPerFrame], state)
		for f := 0; f < n; f++ {
			for c := 0; c < STTConfig.LatentDim; c++ {
				got = append(got, out[c*n+f])
			}
		}
	}
	if len(got) != frames*STTConfig.LatentDim {
		t.Fatalf("streamed %d values, whole gave %d", len(got), frames*STTConfig.LatentDim)
	}
	for f := 0; f < frames; f++ {
		for c := 0; c < STTConfig.LatentDim; c++ {
			a, b := got[f*STTConfig.LatentDim+c], whole[c*frames+f]
			if diff := a - b; diff > 1e-5 || diff < -1e-5 {
				t.Fatalf("frame %d channel %d: streamed %g, whole %g", f, c, a, b)
			}
		}
	}
}
