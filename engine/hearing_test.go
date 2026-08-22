package engine

// The engine speaks to itself.
//
// This test lives here rather than in gemma/ because it needs both engines at
// once — pockettts to say a sentence, Gemma to hear it — and an engine package
// is not allowed to know another one exists. engine/ is the layer above both,
// which is exactly what a test spanning two of them wants.
//
// It is not a fixture comparison. Every other audio test says the tower agrees
// with llama.cpp to some number of decimals; this one is the only one that
// would catch a tower that is numerically right and semantically useless.

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/audio/wav"
	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/pockettts"
)

func TestTheEngineHearsWhatItSaid(t *testing.T) {
	if testing.Short() {
		t.Skip("this test synthesizes and then transcribes; it is not short")
	}
	path, proj := os.Getenv("GOLEM_MODEL"), os.Getenv("GOLEM_MMPROJ")
	if path == "" || proj == "" {
		t.Skip("set GOLEM_MODEL and GOLEM_MMPROJ to run this test")
	}
	const said = "the tower hears what the voice says"
	sound := speak(t, said)

	m, err := Open(path, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.OpenProjector(proj); err != nil {
		t.Fatal(err)
	}
	media, ok := m.Media()
	if !ok || !media.CanHear() {
		t.Skip("this projector carries no audio encoder")
	}

	answer := ask(t, m, media, sound, "Repeat exactly what you hear, and nothing else.", 32)
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
func ask(t *testing.T, m *Model, media Media, sound []byte, question string, most int) string {
	t.Helper()
	rows, err := media.EncodeAudio(sound)
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := m.Template.Render([]chat.Message{{
		Role: "user", Content: question, Audio: [][]byte{sound},
	}}, chat.Options{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	p, err := media.Prompt(m.Vocab.Encode(rendered, false, true), nil, [][][]float32{rows})
	if err != nil {
		t.Fatal(err)
	}
	hidden := media.ForwardPrompt(p, 0)
	last := hidden[len(hidden)-1]
	logits := make([]float32, m.Vocabulary)
	pos := len(p.Tokens)
	var b strings.Builder
	for i := 0; i < most; i++ {
		m.Forward.Logits(last, logits)
		next := argmax(logits)
		if m.Vocab.IsEOG(next) {
			break
		}
		b.WriteString(m.Vocab.Piece(next, false))
		last = m.Forward.ForwardBatch([]int32{next}, pos)[0]
		pos++
	}
	return b.String()
}

func argmax(logits []float32) int32 {
	best := int32(0)
	for i, v := range logits {
		if v > logits[best] {
			best = int32(i)
		}
	}
	return best
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
