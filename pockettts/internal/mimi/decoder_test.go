package mimi

// Parity of the audio decoder: the samples produced frame by frame must match
// those of PyTorch on the same latents.
//
// This is the most revealing test of the set: a streaming-state error does not
// show on the first frame, only at the seam with the second.

import (
	"testing"

	"github.com/ThiraSoft/golem/pockettts/internal/reference"
	"github.com/ThiraSoft/golem/tensors"
)

func TestDecoder(t *testing.T) {
	for _, set := range reference.PipelineSets {
		t.Run(set, func(t *testing.T) { testDecoder(t, set) })
	}
}

func testDecoder(t *testing.T, set string) {
	f := reference.Load(t, set)
	m, err := tensors.Open(reference.ModelPath(t, f.Language))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	d, err := Load(m, DefaultConfig)
	if err != nil {
		t.Fatal(err)
	}

	latents := f.Read(t, "latent")
	want := f.Read(t, "audio")
	L := DefaultConfig.LatentDim

	state := d.NewState(f.Frames)
	for i := 0; i < f.Frames; i++ {
		samples := d.Frame(latents[i*L:(i+1)*L], state)
		if len(samples) != SamplesPerFrame {
			t.Fatalf("frame %d: %d samples, want %d",
				i, len(samples), SamplesPerFrame)
		}
		reference.Compare(t, "audio", samples,
			want[i*SamplesPerFrame:(i+1)*SamplesPerFrame], 1e-3)
	}
}
