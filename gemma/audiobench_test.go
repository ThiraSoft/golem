package gemma

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/audio/decode"
	"github.com/ThiraSoft/golem/audio/mel"
	"github.com/ThiraSoft/golem/tensors"
)

// One recording through the whole conformer, which is what a prompt carrying
// sound pays on top of reading its text. Like the vision tower, it is paid
// once per recording rather than once per token, which is why nothing in it is
// repacked or quantized.
func BenchmarkAudioEncode(b *testing.B) {
	if os.Getenv("GOLEM_MMPROJ") == "" {
		b.Skip("set GOLEM_MMPROJ to run this benchmark")
	}
	benchmarkAudioEncode(b, openMMProj(b))
}

// And the 12B's projector, which is not an encoder: one product against a
// five-megabyte weight, per forty milliseconds of sound.
func BenchmarkUnifiedAudioEncode(b *testing.B) {
	if os.Getenv("GOLEM_MMPROJ_12B") == "" {
		b.Skip("set GOLEM_MMPROJ_12B to run this benchmark")
	}
	benchmarkAudioEncode(b, openMMProj12(b))
}

// The front end alone, which is a fifth of the conformer's cost and all of the
// unified path's: three thousand transforms of 512 points and a filterbank.
func BenchmarkAudioFrontEnd(b *testing.B) {
	samples := benchSpeech(b)
	frames := mel.Frames(len(samples))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mel.Gemma4A(samples)
	}
	b.StopTimer()
	b.ReportMetric(float64(frames), "mel-frames")
}

func benchmarkAudioEncode(b *testing.B, g *tensors.GGUF) {
	cfg, err := LoadAudioConfig(g)
	if err != nil {
		b.Fatal(err)
	}
	w, err := LoadAudioWeights(g, cfg)
	if err != nil {
		b.Fatal(err)
	}
	tower := NewAudioTower(cfg, w)
	samples := benchSpeech(b)

	tokens := len(tower.Encode(samples))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tower.Encode(samples)
	}
	b.StopTimer()
	b.ReportMetric(float64(tokens), "tokens/clip")
	b.ReportMetric(float64(len(samples))/mel.Rate, "seconds-of-audio")
}

func benchSpeech(tb testing.TB) []float32 {
	tb.Helper()
	f, err := os.Open("../testdata/audio/speech.wav")
	if err != nil {
		tb.Skip("the test recording is not on this machine; ref/README.md says how to make it")
	}
	defer f.Close()
	s, rate, ch, err := decode.Decode(f)
	if err != nil {
		tb.Fatal(err)
	}
	if rate != mel.Rate || ch != 1 {
		tb.Fatalf("the fixture is %d Hz on %d channels", rate, ch)
	}
	return s
}
