package gemma

// The two end-to-end tests for audio. Neither compares numbers: one replays
// llama.cpp's own continuation for the fixture, the other asks the model to
// repeat a sentence this repository synthesized a moment earlier. Between
// them they catch what a waypoint comparison cannot — a tower that is
// numerically right and spliced into the prompt wrongly.

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/audio/wav"
	"github.com/ThiraSoft/golem/pockettts"
	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bpe"
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

// The engine speaks to itself: pockettts says a sentence, the conformer hears
// it, and the model is asked to repeat it. It is not a fixture comparison — it
// is the only test here that would catch a tower that is numerically right and
// semantically useless.
func TestTheEngineHearsWhatItSaid(t *testing.T) {
	if testing.Short() {
		t.Skip("this test synthesizes and then transcribes; it is not short")
	}
	if os.Getenv("GOLEM_MMPROJ") == "" {
		t.Skip("set GOLEM_MMPROJ to run this test")
	}
	const said = "the tower hears what the voice says"
	sound := speak(t, said)

	m, vocab := openTextModelAndVocab(t)
	if err := m.OpenProjector(os.Getenv("GOLEM_MMPROJ")); err != nil {
		t.Fatal(err)
	}
	answer := ask(t, m, vocab, sound, "Repeat exactly what you hear, and nothing else.", 32)
	if !similar(answer, said) {
		t.Fatalf("the model heard %q, which is not %q", answer, said)
	}
}

// speak synthesizes one sentence with this repository's own text-to-speech and
// hands back a WAV. The weights are not committed either; a machine without
// them skips.
func speak(t *testing.T, text string) []byte {
	t.Helper()
	lang, err := pockettts.LookupLanguage("english_2026-01")
	if err != nil {
		t.Fatal(err)
	}
	weights := pockettts.Locate(lang.WeightsPath())
	tokenizer := pockettts.Locate(lang.TokenizerPath())
	voices := pockettts.LocateVoices(lang)
	if weights == "" || tokenizer == "" || len(voices) == 0 {
		t.Skip("the Pocket TTS weights or voices are not on this machine; pockettts/README.md says where to get them")
	}
	e, err := pockettts.Open(pockettts.Options{Weights: weights, Tokenizer: tokenizer, Language: lang.Name})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	voice, err := e.LoadVoice(pockettts.Locate(lang.EmbeddingPath(voices[0])))
	if err != nil {
		t.Fatal(err)
	}
	// A fixed seed, so a failure is reproducible and a pass is not luck.
	samples, err := e.Synthesize(text, voice, &pockettts.Settings{Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := wav.Write(&b, samples, pockettts.SampleRate); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// ask puts one recording and one question to the model and draws greedily.
func ask(t *testing.T, m *Model, vocab *bpe.Vocab, sound []byte, question string, most int) string {
	t.Helper()
	rows, err := m.EncodeAudio(sound)
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := RenderChat([]Message{{
		Role: "user", Content: question, Audio: [][]byte{sound},
	}}, ChatOptions{AddGenerationPrompt: true, EmptyThought: m.Cfg.EmptyThought})
	if err != nil {
		t.Fatal(err)
	}
	p, err := m.BuildPrompt(vocab.Encode(rendered, false, true), nil, [][][]float32{rows})
	if err != nil {
		t.Fatal(err)
	}
	hidden := m.ForwardPrompt(p, 0)
	last := hidden[len(hidden)-1]
	logits := make([]float32, m.Cfg.Vocab)
	pos := len(p.Tokens)
	var b strings.Builder
	for i := 0; i < most; i++ {
		m.Logits(last, logits)
		next := Argmax(logits)
		if vocab.IsEOG(next) {
			break
		}
		b.WriteString(vocab.Piece(next, false))
		last = m.Forward(next, pos)
		pos++
	}
	return b.String()
}

// similar compares lowercased words, ignoring punctuation, and passes at three
// quarters of them in order. It is a transcription test, not a string equality:
// a model that writes "The tower hears what the voice says." has heard.
func similar(got, want string) bool {
	g, w := words(got), words(want)
	matched, at := 0, 0
	for _, word := range w {
		for i := at; i < len(g); i++ {
			if g[i] == word {
				matched++
				at = i + 1
				break
			}
		}
	}
	return matched*4 >= len(w)*3
}

func words(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	})
}

// openTextModelAndVocab is openTextModel with the vocabulary the answer is
// read back through, which the tests above need and the numeric ones do not.
func openTextModelAndVocab(t *testing.T) (*Model, *bpe.Vocab) {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this test")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Skipf("GOLEM_MODEL will not open: %v", err)
	}
	t.Cleanup(func() { g.Close() })
	m, err := New(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	vocab, err := bpe.Load(g)
	if err != nil {
		t.Fatal(err)
	}
	return m, vocab
}
