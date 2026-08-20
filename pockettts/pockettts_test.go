package pockettts

// End-to-end test: from text to sound, compared against PyTorch.
//
// Each package checks its own stage; this one checks what links them — the order
// of the operations, the starting latent, the rescaling before the decoder. All
// things no component test can see, and any single one of which is enough to
// make the voice unrecognizable.
//
// The noise comes from the fixtures: that is what makes the comparison possible.

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/pockettts/internal/mimi"
	"github.com/ThiraSoft/golem/pockettts/internal/reference"
	"github.com/ThiraSoft/golem/pockettts/internal/text"
)

// rawText undoes the padding of short inputs, to recover the text a caller
// would actually pass. Everything else prepare_text_prompt does is idempotent.
func rawText(prepared string) string { return strings.TrimLeft(prepared, " ") }

func TestFullSynthesisAgainstReference(t *testing.T) {
	for _, set := range reference.PipelineSets {
		t.Run(set, func(t *testing.T) { testFullSynthesis(t, set) })
	}
}

func testFullSynthesis(t *testing.T, set string) {
	f := reference.Load(t, set)

	engine, err := Open(Options{
		Weights:   reference.ModelPath(t, f.Language),
		Tokenizer: reference.TokenizerPath(t, f.Language),
		Language:  f.Language,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	voice, err := engine.LoadVoice(reference.VoicePath(t, f.Language))
	if err != nil {
		t.Fatal(err)
	}

	// The segmentation must recover the reference's tokens: it is the first
	// thing to diverge when the text preparation is at fault. f.Text is what
	// Python produced *after* preparing it, padding included, so encoding it
	// raw is the right comparison — the padding itself is checked below.
	tokens := engine.tokenizer.Encode(f.Text)
	want := f.Ints(t, "tokens")
	if len(tokens) != len(want) {
		t.Fatalf("%d tokens, want %d", len(tokens), len(want))
	}
	for i := range tokens {
		if tokens[i] != want[i] {
			t.Fatalf("token %d: %d, want %d", i, tokens[i], want[i])
		}
	}

	L := engine.trans.Config.LatentDim
	noises := f.Read(t, "noise")

	// Synthesis is given the *unprepared* text: the engine must apply the
	// language's own rules and land on exactly what Python prepared.
	if got := text.Prepare(rawText(f.Text), engine.rules()); got != f.Text {
		t.Fatalf("prepared text %q, want %q", got, f.Text)
	}

	settings := DefaultSettings(engine.lang)
	// The threshold is disabled: the reference produced a fixed number of frames
	// without stopping at the detected end.
	settings.EndThreshold = 1e9
	settings.noise = func(frame int, into []float32) {
		if frame < f.Frames {
			copy(into, noises[frame*L:(frame+1)*L])
		}
	}
	sound, err := engine.Synthesize(f.Text, voice, &settings)
	if err != nil {
		t.Fatal(err)
	}

	wantAudio := f.Read(t, "audio")
	if len(sound) < len(wantAudio) {
		t.Fatalf("%d samples produced, the reference has %d", len(sound), len(wantAudio))
	}
	// The tolerance is wider than that of the component tests, and it has a
	// reason: generation is a loop. The latent of one frame is fed back into the
	// next, so that the rounding gap does not repeat, it accumulates — from 5e-6
	// on the first frame to 3e-4 on the eighth. It is numerical drift, not a
	// divergence of computation: the stages, tested in isolation, stay at 5e-7.
	for i := 0; i < f.Frames; i++ {
		start, end := i*mimi.SamplesPerFrame, (i+1)*mimi.SamplesPerFrame
		reference.Compare(t, "audio", sound[start:end], wantAudio[start:end], 5e-3)
	}
}

func TestDefaultSettings(t *testing.T) {
	lang, err := LookupLanguage("french_24l")
	if err != nil {
		t.Fatal(err)
	}
	s := DefaultSettings(lang)
	if s.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", s.Temperature)
	}
	if s.EndThreshold != -4 {
		t.Errorf("EndThreshold = %v, want -4", s.EndThreshold)
	}
	if s.FramesAfterEnd != lang.FramesAfterEnd {
		t.Errorf("FramesAfterEnd = %v, want %v", s.FramesAfterEnd, lang.FramesAfterEnd)
	}
	if s.MaxTokens != 50 {
		t.Errorf("MaxTokens = %v, want 50", s.MaxTokens)
	}
}

// A caller that means zero gets zero. That is the whole point: gladyss sets an
// end threshold of 0.0 on purpose, and the old zero-means-default rule turned
// it into -4 without saying so.
func TestSettingsKeepsAnExplicitZero(t *testing.T) {
	lang, err := LookupLanguage("french_24l")
	if err != nil {
		t.Fatal(err)
	}
	s := DefaultSettings(lang)
	s.EndThreshold = 0
	if s.EndThreshold != 0 {
		t.Errorf("EndThreshold = %v, want 0", s.EndThreshold)
	}
}

// testEngine opens the French engine on the weights in the Hugging Face cache,
// and loads the first voice it finds. It skips when the model is not there:
// a fresh clone has no weights, and these tests are about the engine's control
// flow, not about anything a fixture could stand in for.
func testEngine(t *testing.T) (*Engine, *Voice) {
	t.Helper()
	lang, err := LookupLanguage(DefaultLanguage)
	if err != nil {
		t.Fatal(err)
	}
	weights, tokenizer := Locate(lang.WeightsPath()), Locate(lang.TokenizerPath())
	if weights == "" || tokenizer == "" {
		t.Skip("Pocket TTS weights not in the Hugging Face cache")
	}
	engine, err := Open(Options{Weights: weights, Tokenizer: tokenizer, Language: lang.Name})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { engine.Close() })

	var voice *Voice
	for _, name := range LocateVoices(lang) {
		if voice, err = engine.LoadVoice(Locate(lang.EmbeddingPath(name))); err == nil {
			break
		}
		t.Fatal(err)
	}
	if voice == nil {
		t.Skip("no voice in the Hugging Face cache")
	}
	return engine, voice
}

// A cancelled context stops the generation rather than running the text to its
// end.
func TestSynthesizeStopsOnCancel(t *testing.T) {
	engine, voice := testEngine(t)

	ctx, cancel := context.WithCancel(context.Background())
	settings := DefaultSettings(engine.lang)
	settings.Seed = 1
	settings.Frame = func([]float32) { cancel() } // stop at the very first frame
	settings.Ctx = ctx

	_, err := engine.Synthesize("Une phrase assez longue pour occuper le modèle un moment.", voice, &settings)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

// Without a context, nothing changes.
func TestSynthesizeWithoutContext(t *testing.T) {
	engine, voice := testEngine(t)

	settings := DefaultSettings(engine.lang)
	settings.Seed = 1
	sound, err := engine.Synthesize("Bonjour.", voice, &settings)
	if err != nil {
		t.Fatal(err)
	}
	if len(sound) == 0 {
		t.Fatal("no sound produced")
	}
}
