package stt

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/internal/kyutai/reference"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

func sttWeights(t *testing.T) *tensors.Model {
	t.Helper()
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set: the STT checkpoint is not on this machine")
	}
	m, err := tensors.Open(dir + "/model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestBlockZeroAgainstReference(t *testing.T) {
	f := reference.Load(t, "stt")
	w, err := LoadWeights(sttWeights(t), nn.BF16)
	if err != nil {
		t.Fatal(err)
	}
	emb := f.Read(t, "emb")     // [1, frames, DModel]
	want := f.Read(t, "block0") // same shape
	frames := len(emb) / DModel
	kv := NewKV()
	scratch := NewScratch()
	got := make([]float32, 0, len(emb))
	x := make([]float32, DModel)
	for i := 0; i < frames; i++ {
		copy(x, emb[i*DModel:(i+1)*DModel])
		w.Layers[0].Step(x, kv[0], scratch)
		got = append(got, x...)
	}
	reference.Compare(t, "block0", got, want, 5e-5)
}

func TestLoadWeightsAndStep(t *testing.T) {
	w, err := LoadWeights(sttWeights(t), nn.BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Layers) != NumLayers {
		t.Fatalf("layers = %d, want %d", len(w.Layers), NumLayers)
	}
	if len(w.Audio) != Codebooks {
		t.Fatalf("audio codebooks = %d, want %d", len(w.Audio), Codebooks)
	}
	kv := NewKV()
	scratch := NewScratch()
	x := make([]float32, DModel)
	for i := range x {
		x[i] = float32(i%10) * 0.1
	}
	w.Layers[0].Step(x, kv[0], scratch)
	if kv[0].Position != 1 {
		t.Fatalf("kv.Position = %d, want 1", kv[0].Position)
	}
}

func TestTrunkAndLogitsAgainstReference(t *testing.T) {
	f := reference.Load(t, "stt")
	w, err := LoadWeights(sttWeights(t), nn.BF16)
	if err != nil {
		t.Fatal(err)
	}
	emb, wantTrunk, wantLogits := f.Read(t, "emb"), f.Read(t, "trunk"), f.Read(t, "logits")
	frames := len(emb) / DModel
	kv := NewKV()
	scratch := NewScratch()
	trunk := make([]float32, 0, len(emb))
	logits := make([]float32, 0, frames*TextCard)
	x := make([]float32, DModel)
	row := make([]float32, TextCard)
	for i := 0; i < frames; i++ {
		copy(x, emb[i*DModel:(i+1)*DModel])
		for l, layer := range w.Layers {
			layer.Step(x, kv[l], scratch)
		}
		nn.RMSNormPlain(x, w.OutNorm, NormEps)
		trunk = append(trunk, x...)
		product(w.Head, scratch.wide, x, row)
		logits = append(logits, row...)
	}
	// Five parts in ten thousand, and the number is measured rather than
	// chosen: sixteen layers of float32 residual accumulate 2e-4 of the
	// tensor's own scale against PyTorch, which sums in a different order and
	// fuses different multiplies. The single waypoints above hold at 5e-5,
	// which is where a fault would show. This was 1.32 — a hundred and
	// thirty-two per cent — until the norms' epsilon was found to be 1e-8.
	reference.Compare(t, "trunk", trunk, wantTrunk, 5e-4)
	reference.Compare(t, "logits", logits, wantLogits, 5e-5)
}

func TestWholeTrunkSynthetic(t *testing.T) {
	w, err := LoadWeights(sttWeights(t), nn.BF16)
	if err != nil {
		t.Fatal(err)
	}
	frames := 4
	kv := NewKV()
	scratch := NewScratch()
	x := make([]float32, DModel)
	row := make([]float32, TextCard)
	for i := 0; i < frames; i++ {
		for j := range x {
			x[j] = float32(i+j%10) * 0.01
		}
		for l, layer := range w.Layers {
			layer.Step(x, kv[l], scratch)
		}
		nn.RMSNormPlain(x, w.OutNorm, NormEps)
		product(w.Head, scratch.wide, x, row)
		maxVal, maxIdx := float32(-1e9), -1
		for idx, v := range row {
			if v > maxVal {
				maxVal, maxIdx = v, idx
			}
		}
		if maxIdx < 0 || maxIdx >= TextCard {
			t.Fatalf("frame %d: invalid max token %d", i, maxIdx)
		}
	}
}
