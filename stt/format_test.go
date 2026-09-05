package stt

// What each weight format costs the transcript.
//
// The cosine of a product tells you the format's noise floor and not what it
// does to a word. This does: the same speech through bfloat16, Q8_0 and Q4_0,
// judged by the word error rate against what is actually said. It is the only
// test here entitled to an opinion on which format ships.

import (
	"os"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/nn"
)

func TestFormatsAgreeOnTheTranscript(t *testing.T) {
	heavy.Skip(t, "it transcribes real speech end to end on the processor")
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set")
	}
	clip, err := os.ReadFile("../testdata/stt/clip.wav")
	if err != nil {
		t.Skipf("no clip: %v", err)
	}
	spoken, err := os.ReadFile("../testdata/stt/transcript.txt")
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Fields(string(spoken))

	for _, quant := range []nn.Quant{nn.BF16, nn.Q8_0, nn.Q4_0} {
		t.Run(quant.String(), func(t *testing.T) {
			o, err := Locate(dir)
			if err != nil {
				t.Fatal(err)
			}
			o.Quant = quant
			m, err := Open(o)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			got, err := m.Transcribe(clip)
			if err != nil {
				t.Fatal(err)
			}
			rate := wordErrorRate(want, strings.Fields(got))
			t.Logf("%s: %.3f word error rate — %q", quant, rate, got)
			if rate > 0.15 {
				t.Errorf("%s: word error rate %.2f", quant, rate)
			}
		})
	}
}

// TestFormatsOnRealSpeech is the discriminating one: half a minute of read
// English with proper nouns, dates and a headline in it, where a format that
// costs something will cost it on a name rather than on "bonjour". bfloat16 is
// the reference — there is no ground truth file for this clip, and what is
// being measured is what quantizing takes away, not what the model knows.
func TestFormatsOnRealSpeech(t *testing.T) {
	heavy.Skip(t, "it transcribes real speech end to end on the processor")
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set")
	}
	clip, err := os.ReadFile("../testdata/audio/speech.wav")
	if err != nil {
		t.Skipf("no speech.wav: %v", err)
	}

	got := map[nn.Quant]string{}
	for _, quant := range []nn.Quant{nn.BF16, nn.Q8_0, nn.Q4_0} {
		o, err := Locate(dir)
		if err != nil {
			t.Fatal(err)
		}
		o.Quant = quant
		m, err := Open(o)
		if err != nil {
			t.Fatal(err)
		}
		text, err := m.Transcribe(clip)
		m.Close()
		if err != nil {
			t.Fatal(err)
		}
		got[quant] = text
		t.Logf("%s: %q", quant, text)
	}

	reference := strings.Fields(got[nn.BF16])

	// Q8_0 is the default, and what makes it the default is that it costs
	// nothing at all: word for word what bfloat16 says. That is the assertion.
	if rate := wordErrorRate(reference, strings.Fields(got[nn.Q8_0])); rate != 0 {
		t.Errorf("Q8_0 drifts from bfloat16 by %.4f over %d words:\n got: %s\nwant: %s",
			rate, len(reference), got[nn.Q8_0], got[nn.BF16])
	}

	// Q4_0 does cost something, and this records how much rather than
	// pretending otherwise: half again the speed for six per cent of the
	// words, which on this clip is a date losing its "st" and "newsprint and
	// ink" becoming "news printing ink". The bound is there to catch a
	// regression, not to bless the number.
	rate := wordErrorRate(reference, strings.Fields(got[nn.Q4_0]))
	t.Logf("Q4_0 against bfloat16: %.4f word error rate over %d words", rate, len(reference))
	if rate > 0.10 {
		t.Errorf("Q4_0 drifts from bfloat16 by %.2f, which is more than it used to", rate)
	}
}
