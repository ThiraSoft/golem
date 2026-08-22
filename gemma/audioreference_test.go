package gemma

// Reading what llama.cpp recorded for one piece of audio. Neither the
// recording nor the sound file it was made from is committed —
// ref/README.md writes both in one command each — and a machine without them
// skips.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/audio/decode"
	"github.com/ThiraSoft/golem/audio/mel"
	"github.com/ThiraSoft/golem/tensors"
)

// audioFixture is the recording plus what the text side has to agree with:
// where the recording sits in the prompt and how many tokens it became.
type audioFixture struct {
	*fixture
	AudioStart   int `json:"audio_start"`
	NAudioTokens int `json:"n_audio_tokens"`
}

// loadAudioFixture reads one of the audio recordings: "audio" for E2B's
// conformer, "audio12" for the 12B's projection.
func loadAudioFixture(t *testing.T, name string) *audioFixture {
	t.Helper()
	f := loadFixture(t, name)
	a := &audioFixture{fixture: f}
	raw, err := os.ReadFile(filepath.Join(f.dir, "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, a); err != nil {
		t.Fatal(err)
	}
	return a
}

// Frames is the number of positions the tower works on: the recording's own
// second dimension, read from whatever waypoint is asked for.
func (a *audioFixture) frames(t *testing.T, name string) int {
	t.Helper()
	e, ok := a.Tensors[name]
	if !ok {
		t.Fatalf("the fixture holds no tensor named %q", name)
	}
	return int(e.NE[1])
}

// melInput reads the mel the reference preprocessor built and turns it around.
// The recorder kept the graph's own first node, which is [128 mel, frames]
// with the frequency the faster index; the front end here produces the
// transpose of that, and so does everything downstream of it.
func (a *audioFixture) melInput(t *testing.T) ([]float32, int) {
	t.Helper()
	raw := a.tensor(t, "mel")
	bins := int(a.Tensors["mel"].NE[0])
	frames := int(a.Tensors["mel"].NE[1])
	out := make([]float32, len(raw))
	for f := 0; f < frames; f++ {
		for m := 0; m < bins; m++ {
			out[m*frames+f] = raw[f*bins+m]
		}
	}
	return out, frames
}

// speechSamples is the recording the fixture was made from, decoded.
func loadSpeech(t *testing.T) []float32 {
	t.Helper()
	path := filepath.Join(repoRoot(t), "testdata", "audio", "speech.wav")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("%s is not here; ref/README.md says how to make it", path)
	}
	defer f.Close()
	s, rate, ch, err := decode.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	if rate != mel.Rate || ch != 1 {
		t.Fatalf("the fixture is %d Hz on %d channels, not 16 kHz mono", rate, ch)
	}
	return s
}

// openAudioTower builds the conformer from the projector named by GOLEM_MMPROJ.
func openAudioTower(t *testing.T) *AudioTower {
	t.Helper()
	return audioTowerFrom(t, openMMProj(t))
}

// openAudioTower12 builds the 12B's projection from GOLEM_MMPROJ_12B.
func openAudioTower12(t *testing.T) *AudioTower {
	t.Helper()
	return audioTowerFrom(t, openMMProj12(t))
}

func audioTowerFrom(t *testing.T, g *tensors.GGUF) *AudioTower {
	t.Helper()
	cfg, err := LoadAudioConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadAudioWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return NewAudioTower(cfg, w)
}
