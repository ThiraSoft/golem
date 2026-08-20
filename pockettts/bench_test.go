package pockettts

// What the engine costs, end to end.
//
// The unit that matters for speech is not the operation but the second: a
// synthesizer is fast enough when it produces sound faster than the sound is
// spoken, and everything above that is headroom for a slower machine or a
// second speaker. So the benchmark reports the ratio — seconds of audio per
// second of wall clock — beside the usual per-operation time.
//
// One synthesis is one sentence, seeded, so that two runs draw the same noise
// and produce the same number of frames. Without that the model decides for
// itself when the sentence is over, and the measurement moves with it.

import (
	"testing"
	"time"

	"github.com/ThiraSoft/golem/pockettts/internal/reference"
)

// One sentence per language, each long enough to leave the startup behind and
// short enough to stay inside one segment. The two languages are two models,
// not one model reading two texts.
var benchLanguages = []struct{ language, sentence string }{
	{"french_24l", "Le vent se lève, il faut tenter de vivre, et la mer scintille au loin."},
	{"english_2026-01", "The wind is rising, we must try to live, and the sea glitters far away."},
}

func benchEngine(b *testing.B, language string) (*Engine, *Voice) {
	b.Helper()
	engine, err := Open(Options{
		Weights:   reference.ModelPath(b, language),
		Tokenizer: reference.TokenizerPath(b, language),
		Language:  language,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { engine.Close() })

	voice, err := engine.LoadVoice(reference.VoicePath(b, language))
	if err != nil {
		b.Fatal(err)
	}
	return engine, voice
}

// BenchmarkSynthesis is the whole path: text in, samples out.
func BenchmarkSynthesis(b *testing.B) {
	for _, l := range benchLanguages {
		b.Run(l.language, func(b *testing.B) { benchmarkSynthesis(b, l.language, l.sentence) })
	}
}

func benchmarkSynthesis(b *testing.B, language, sentence string) {
	engine, voice := benchEngine(b, language)
	set := DefaultSettings(engine.lang)
	set.Seed = 20260820
	settings := &set

	// One synthesis before the clock starts: the first frame of a run pays for
	// the pages of the mapping it touches first, which is startup and not
	// speed.
	if _, err := engine.Synthesize(sentence, voice, settings); err != nil {
		b.Fatal(err)
	}

	var samples int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sound, err := engine.Synthesize(sentence, voice, settings)
		if err != nil {
			b.Fatal(err)
		}
		samples += len(sound)
	}
	b.StopTimer()

	seconds := float64(samples) / SampleRate
	b.ReportMetric(seconds/b.Elapsed().Seconds(), "x-real-time")
	b.ReportMetric(seconds/float64(b.N), "audio-s/op")
}

// BenchmarkFirstFrame is the other number a speaking machine is judged on: how
// long the silence lasts before there is something to play. The loop still
// synthesizes the whole sentence — what is reported is the moment the first
// frame arrived, not what the sentence cost.
func BenchmarkFirstFrame(b *testing.B) {
	for _, l := range benchLanguages {
		b.Run(l.language, func(b *testing.B) { benchmarkFirstFrame(b, l.language, l.sentence) })
	}
}

func benchmarkFirstFrame(b *testing.B, language, sentence string) {
	engine, voice := benchEngine(b, language)
	warmup := DefaultSettings(engine.lang)
	warmup.Seed = 20260820
	if _, err := engine.Synthesize(sentence, voice, &warmup); err != nil {
		b.Fatal(err)
	}

	var total time.Duration
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var first time.Duration
		start := time.Now()
		settings := DefaultSettings(engine.lang)
		settings.Seed = 20260820
		settings.Frame = func([]float32) {
			if first == 0 {
				first = time.Since(start)
			}
		}
		if _, err := engine.Synthesize(sentence, voice, &settings); err != nil {
			b.Fatal(err)
		}
		if first == 0 {
			b.Fatal("no frame was produced")
		}
		total += first
	}
	b.StopTimer()
	b.ReportMetric(float64(total)/float64(b.N)/1e6, "ms-to-first-frame")
}
