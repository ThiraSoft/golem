package gemma

// The end-to-end check for audio inside this engine: llama.cpp's own prompt,
// rebuilt here, reaching the same answer.
//
// Its companion lives in engine/, not here. That one has this repository's
// text-to-speech say a sentence and asks the model to repeat it, which is the
// only test that would catch a tower that is numerically right and
// semantically useless — and it needs both engines at once, which no engine
// package is allowed to do.

import (
	"os"
	"testing"
)

// The prompt this engine builds reaches llama.cpp's own first token.
//
// Only the first, and that is deliberate. Replaying the whole recorded
// continuation — which is what the vision side does — says nothing here: give
// this model llama.cpp's own projected rows, byte for byte, and it still parts
// company at the fourth token by two and a half logits. The prompt is four
// hundred and fifty tokens long against the picture test's forty, the
// checkpoint is Q4_0, and the question is open enough that "the speaker is
// expressing" and "the speaker is comparing" are a genuine fork. What the
// tower produces is pinned by the waypoint tests; what the model does with a
// long quantized prompt afterwards is not this test's business.
//
// The sentence the model actually heard is checked by
// TestTheEngineHearsWhatItSaid below, which is the honest end-to-end test.
func TestAudioPromptReachesTheSameFirstToken(t *testing.T) {
	f := loadAudioFixture(t, "audio")
	if os.Getenv("GOLEM_MMPROJ") == "" {
		t.Skip("set GOLEM_MMPROJ to run this test")
	}
	m := openTextModel(t)
	if err := m.OpenProjector(os.Getenv("GOLEM_MMPROJ")); err != nil {
		t.Fatal(err)
	}
	rows, err := m.EncodeAudio(speechBytes(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != f.NAudioTokens {
		t.Fatalf("this engine made %d audio tokens, llama.cpp %d", len(rows), f.NAudioTokens)
	}

	// The prompt is the fixture's own text tokens, which keeps this test about
	// the recording rather than about the template. The recorder wrote the
	// padding token where the soft tokens went, so what is left after taking
	// them out is the text, markers included, and BuildPrompt fills the
	// markers back in.
	var text []int32
	text = append(text, f.Tokens[:f.AudioStart]...)
	text = append(text, f.Tokens[f.AudioStart+f.NAudioTokens:]...)
	p, err := m.BuildPrompt(text, nil, [][][]float32{rows})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Tokens) != len(f.Tokens) {
		t.Fatalf("the prompt came to %d tokens, llama.cpp's to %d", len(p.Tokens), len(f.Tokens))
	}
	for i := range p.Tokens {
		if p.Embeds[i] == nil && p.Tokens[i] != f.Tokens[i] {
			t.Fatalf("token %d is %d, llama.cpp's is %d", i, p.Tokens[i], f.Tokens[i])
		}
	}

	hidden := m.ForwardPrompt(p, 0)
	logits := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden[len(hidden)-1], logits)
	if got := Argmax(logits); got != f.Argmax {
		t.Fatalf("this engine answers with %d, llama.cpp with %d (by %v)",
			got, f.Argmax, logits[got]-logits[f.Argmax])
	}
}
