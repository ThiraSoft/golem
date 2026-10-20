package main

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ThiraSoft/golem/audio/wav"
	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/yue2"
)

// TestGuerrerosReplay renders the whole recorded replay as the command does,
// at the model's whole context (one chunk of 4870 frames, as yue-synth), and
// holds it to yue-synth's render of the same request: the normalized wav16
// against bench/q8-replay.wav, and the replay request against
// bench/q8-replay.json, field for field. It lives beside the command rather
// than in yue2 so that it has the card to itself: the pipeline takes 13.45 GB
// at that context, and yue2's tests keep their own halves resident.
func TestGuerrerosReplay(t *testing.T) {
	heavy.Skip(t, "a whole song of 4870 frames, 13.45 GB on the card")
	bench := filepath.Join("..", "..", "testdata", "yue2", "bench")
	r, err := yue2.ReadRequest(filepath.Join(bench, "guerreros-replay.json"))
	if err != nil {
		t.Skipf("no recorded replay (%v): see ref/yue2/README.md", err)
	}
	want, err := yue2.ReadRequest(filepath.Join(bench, "q8-replay.json"))
	if err != nil {
		t.Skipf("no recorded replay (%v): see ref/yue2/README.md", err)
	}
	f, err := os.Open(filepath.Join(bench, "q8-replay.wav"))
	if err != nil {
		t.Skipf("no recorded render (%v): see ref/yue2/README.md", err)
	}
	ref, rate, channels, err := wav.ReadChannels(f)
	f.Close()
	if err != nil || rate != yue2.SampleRate || channels != 2 {
		t.Fatalf("q8-replay.wav: %v, %d Hz, %d channels", err, rate, channels)
	}

	p, err := yue2.Open(yue2.Options{})
	if err != nil {
		t.Skipf("no pipeline: %v", err)
	}
	defer p.Close()
	song, err := p.Generate(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d frames, times %v", song.Frames, song.Times)

	b, err := song.Encode(r.OutputFormat, r.PeakClip)
	if err != nil {
		t.Fatal(err)
	}
	got, _, _, err := wav.ReadChannels(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(ref) {
		t.Fatalf("%d samples, yue-synth wrote %d", len(got), len(ref))
	}
	var num, den float64
	for i := range ref {
		d := float64(got[i] - ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	snr := 10 * math.Log10(den/num)
	t.Logf("normalized wav16 against q8-replay.wav: SNR %.2f dB", snr)
	if !(snr >= 30) {
		t.Errorf("normalized wav16: SNR %.2f dB below 30 dB", snr)
	}

	back, err := yue2.ParseRequest(song.Replay(r).JSON())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, want) {
		t.Errorf("the replay request differs from yue-synth's:\n%s", song.Replay(r).JSON())
	}
}
