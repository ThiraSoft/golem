package flowlm

// Parity of the flow net. The starting noise comes from the fixtures: that is
// what makes the comparison possible, generation being random by nature.

import (
	"testing"

	"github.com/ThiraSoft/golem/pockettts/internal/reference"
	"github.com/ThiraSoft/golem/tensors"
)

func TestFlowNet(t *testing.T) {
	for _, set := range reference.PipelineSets {
		t.Run(set, func(t *testing.T) { testFlowNet(t, set) })
	}
}

func testFlowNet(t *testing.T, set string) {
	f := reference.Load(t, set)
	m, err := tensors.Open(reference.ModelPath(t, f.Language))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	cfg := ConfigFor(f.NumLayers)
	r, err := LoadFlowNet(m, cfg)
	if err != nil {
		t.Fatal(err)
	}

	D, L := cfg.DModel, cfg.LatentDim
	cond := f.Read(t, "cond")
	noise := f.Read(t, "noise")
	want := f.Read(t, "latent")

	got := make([]float32, L)
	for i := 0; i < f.Frames; i++ {
		r.Latent(cond[i*D:(i+1)*D], noise[i*L:(i+1)*L], 1, got)
		reference.Compare(t, "latent", got, want[i*L:(i+1)*L], 5e-3)
	}
}
