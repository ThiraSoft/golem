package flowlm

// Parity of the full transformer: the 24 layers, on the real voice state.
//
// This test checks the voice import at the same time — if the K/V caches were
// laid out wrong, the prompt output would diverge from the first position.

import (
	"testing"

	"github.com/ThiraSoft/golem/pockettts/internal/reference"
	"github.com/ThiraSoft/golem/tensors"
)

func loadTransformer(t *testing.T, f reference.Fixtures) *Transformer {
	t.Helper()
	m, err := tensors.Open(reference.ModelPath(t, f.Language))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	tr, err := Load(m, ConfigFor(f.NumLayers))
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

// TestTextPrompt checks the output of the layers on the text tokens, starting
// from the voice state. It runs for every language that has a fixture set: a
// 24-layer model and a 6-layer one exercise the same code.
func TestTextPrompt(t *testing.T) {
	for _, set := range reference.PipelineSets {
		t.Run(set, func(t *testing.T) { testTextPrompt(t, set) })
	}
}

func testTextPrompt(t *testing.T, set string) {
	f := reference.Load(t, set)
	tr := loadTransformer(t, f)

	state, err := tr.LoadVoice(reference.VoicePath(t, f.Language), f.VoiceOffset+64)
	if err != nil {
		t.Fatal(err)
	}
	if state.Position() != f.VoiceOffset {
		t.Fatalf("voice loaded at position %d, want %d", state.Position(), f.VoiceOffset)
	}

	tokens := f.Ints(t, "tokens")
	D := tr.Config.DModel
	wantEmb := f.Read(t, "text_emb")
	want := f.Read(t, "prompt_out")

	for i, tok := range tokens {
		reference.Compare(t, "text_emb", tr.Conditioner[tok], wantEmb[i*D:(i+1)*D], 5e-3)
		out := tr.AdvanceText(tok, state)
		reference.Compare(t, "prompt_out", out, want[i*D:(i+1)*D], 5e-3)
	}
}

// TestConditioning carries on into generation: each frame feeds back the
// reference latent, which isolates the transformer from the flow net.
func TestConditioning(t *testing.T) {
	for _, set := range reference.PipelineSets {
		t.Run(set, func(t *testing.T) { testConditioning(t, set) })
	}
}

func testConditioning(t *testing.T, set string) {
	f := reference.Load(t, set)
	tr := loadTransformer(t, f)

	state, err := tr.LoadVoice(reference.VoicePath(t, f.Language), f.VoiceOffset+64)
	if err != nil {
		t.Fatal(err)
	}
	for _, tok := range f.Ints(t, "tokens") {
		tr.AdvanceText(tok, state)
	}

	D, L := tr.Config.DModel, tr.Config.LatentDim
	wantCond := f.Read(t, "cond")
	wantEOS := f.Read(t, "eos_logit")
	latents := f.Read(t, "latent")

	latent := append([]float32(nil), tr.BOSEmb...) // the first frame starts from the BOS
	for i := 0; i < f.Frames; i++ {
		cond := tr.AdvanceLatent(latent, state)
		reference.Compare(t, "cond", cond, wantCond[i*D:(i+1)*D], 5e-3)

		eos := tr.EOSLogit(cond)
		if gap := eos - wantEOS[i]; gap > 1e-2 || gap < -1e-2 {
			t.Errorf("eos[%d] = %.6f, want %.6f", i, eos, wantEOS[i])
		}
		latent = latents[i*L : (i+1)*L]
	}
}
