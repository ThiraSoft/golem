package decode

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestFormatsAgree decodes the same half second of tone from three files and
// expects the same signal back. The three files are not committed; a machine
// without them skips, and makes them with:
//
//	mkdir -p testdata/audio
//	ffmpeg -f lavfi -i "sine=frequency=440:duration=0.5:sample_rate=16000" \
//	    -af volume=6 -ac 1 testdata/audio/tone.wav
//	ffmpeg -i testdata/audio/tone.wav testdata/audio/tone.mp3
//	ffmpeg -i testdata/audio/tone.wav testdata/audio/tone.flac
//
// The tone is amplified because lavfi writes it at an eighth of full scale,
// quiet enough that the MP3 encoder loses a twentieth of its energy.
//
// The channel counts are not expected to agree: a mono MP3 still comes out of
// its decoder as two identical channels, because that is what the decoder
// hands back. What must agree is the sound, once the channels are averaged.
func TestFormatsAgree(t *testing.T) {
	dir := filepath.Join("..", "..", "testdata", "audio")
	ref, rate, ch := mustDecode(t, filepath.Join(dir, "tone.wav"))
	if rate != 16000 || ch != 1 {
		t.Fatalf("the reference tone is %d Hz on %d channels, not 16 kHz mono", rate, ch)
	}
	for _, name := range []string{"tone.mp3", "tone.flac"} {
		got, r, c := mustDecode(t, filepath.Join(dir, name))
		if r != rate {
			t.Fatalf("%s: %d Hz, want %d", name, r, rate)
		}
		// Lossy formats pad and ring; compare the energy of the middle, not
		// the samples, and only for mp3.
		if err := closeEnough(ref, mono(got, c), name == "tone.mp3"); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

func TestFormatIsReadFromTheHeader(t *testing.T) {
	for head, want := range map[string]string{
		"RIFF\x00\x00\x00\x00WAVE": "wav",
		"fLaC\x00\x00\x00\x22":     "flac",
		"ID3\x04\x00\x00\x00\x00":  "mp3",
		"\xff\xfb\x90\x64\x00\x00": "mp3",
		"not audio at all":         "",
	} {
		if got := Format([]byte(head)); got != want {
			t.Errorf("Format(%q) = %q, want %q", head, got, want)
		}
	}
}

// TestBytesThatAreNoFormatSaySo: an unrecognised file is a sentence, not a
// silent empty signal.
func TestBytesThatAreNoFormatSaySo(t *testing.T) {
	if _, _, _, err := Decode(strings.NewReader("this is a text file, not a sound")); err == nil {
		t.Fatal("Decode accepted bytes that are no audio format")
	}
	if _, _, _, err := Decode(strings.NewReader("")); err == nil {
		t.Fatal("Decode accepted an empty file")
	}
}

func mustDecode(t *testing.T, path string) ([]float32, int, int) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("%s is not here; the ffmpeg lines above TestFormatsAgree make it", path)
	}
	defer f.Close()
	s, rate, ch, err := Decode(f)
	if err != nil {
		t.Fatalf("decoding %s: %v", path, err)
	}
	return s, rate, ch
}

// mono averages an interleaved signal down to one channel.
func mono(s []float32, channels int) []float32 {
	if channels == 1 {
		return s
	}
	out := make([]float32, len(s)/channels)
	for i := range out {
		var sum float32
		for c := 0; c < channels; c++ {
			sum += s[i*channels+c]
		}
		out[i] = sum / float32(channels)
	}
	return out
}

// closeEnough compares two signals: by the energy of their middle for a lossy
// codec, sample for sample otherwise.
func closeEnough(ref, got []float32, lossy bool) error {
	if !lossy {
		if len(got) != len(ref) {
			return fmt.Errorf("%d samples, want %d", len(got), len(ref))
		}
		for i := range ref {
			if d := math.Abs(float64(got[i] - ref[i])); d > 1e-4 {
				return fmt.Errorf("sample %d is %f away from the reference", i, d)
			}
		}
		return nil
	}
	a, b := rms(ref), rms(got)
	if d := math.Abs(float64(a-b)) / float64(a); d > 0.05 {
		return fmt.Errorf("the middle carries %.4f of energy against the reference's %.4f", b, a)
	}
	return nil
}

// rms is the energy of the fifth of a signal that sits around its middle. A
// codec pads: an MP3 of half a second of tone comes back nearly two thousand
// samples longer than it went in, silence at both ends. The middle is the part
// both files certainly hold the same sound in.
func rms(s []float32) float32 {
	from, to := 2*len(s)/5, 3*len(s)/5
	var sum float64
	for _, v := range s[from:to] {
		sum += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sum / float64(to-from)))
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
