package mel

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/ThiraSoft/golem/audio/decode"
)

// The mel the reference preprocessor built for the fixture, sample for
// sample. This is the test that says the paddings are right: the semicausal
// left padding, the right padding to PyTorch's frame count, and the trim back
// to it.
func TestGemma4AMatchesTheRecording(t *testing.T) {
	want, frames := loadRecordedMel(t)
	samples := loadSpeech(t)
	got := Gemma4A(samples)
	if len(got) != 1 {
		t.Fatalf("the fixture is under thirty seconds and gave %d chunks", len(got))
	}
	if len(got[0]) != len(want) {
		t.Fatalf("mel of %d values over %d frames, recording has %d", len(got[0]), Frames(len(samples)), len(want))
	}
	// Two bounds rather than one. The reference computes its transform in
	// float32 and takes a logarithm afterwards, so a bin where the frequencies
	// nearly cancel comes back with a relative error the log turns into about a
	// thousandth: a handful of the two hundred thousand values here are that
	// far off, and demanding 1e-4 of all of them would be demanding llama.cpp's
	// rounding rather than its arithmetic. The bulk is held tight, which is
	// what catches a padding off by a frame — that would move everything, not
	// a hundred values.
	d := make([]float64, len(want))
	for i := range want {
		d[i] = math.Abs(float64(got[0][i] - want[i]))
	}
	sort.Float64s(d)
	if p99 := d[len(d)*99/100]; p99 > 1e-4 {
		t.Fatalf("over %d frames, a hundredth of the mel is more than %.6f away from llama.cpp's", frames, p99)
	}
	if worst := d[len(d)-1]; worst > 2e-3 {
		t.Fatalf("the mel over %d frames is %.6f away from llama.cpp's", frames, worst)
	}
}

// TestFramesCountsWhatGemma4AProduces: the tower sizes its scratch from this
// before the mel exists, so the two have to agree.
func TestFramesCountsWhatGemma4AProduces(t *testing.T) {
	for _, n := range []int{1600, 16000, 16000*30 + 5, 320} {
		want := 0
		for _, chunk := range Gemma4A(make([]float32, n)) {
			want += len(chunk) / Bins
		}
		if got := Frames(n); got != want {
			t.Errorf("Frames(%d) = %d, Gemma4A gave %d", n, got, want)
		}
	}
}

// loadRecordedMel reads testdata/gemma/audio/mel.bin, which the recorder wrote
// as [128 mel, frames] with the frequency the faster index — the transpose of
// what Gemma4A returns, so it is turned around here.
func loadRecordedMel(t *testing.T) ([]float32, int) {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "gemma", "audio", "mel.bin")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("%s is not here; ref/README.md says how to record it", path)
	}
	n := len(raw) / 4
	if n%Bins != 0 {
		t.Fatalf("%s holds %d floats, not a multiple of %d mel bins", path, n, Bins)
	}
	frames := n / Bins
	out := make([]float32, n)
	for t := 0; t < frames; t++ {
		for m := 0; m < Bins; m++ {
			out[m*frames+t] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(t*Bins+m):]))
		}
	}
	return out, frames
}

func loadSpeech(t *testing.T) []float32 {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "audio", "speech.wav")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("%s is not here; ref/README.md says how to make it", path)
	}
	defer f.Close()
	s, rate, ch, err := decode.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 || ch != 1 {
		t.Fatalf("the fixture is %d Hz on %d channels, not 16 kHz mono", rate, ch)
	}
	return s
}
