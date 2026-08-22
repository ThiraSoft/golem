package gemma

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

// openMMProj26B is mmproj26BPath opened, for the one test that asks a
// projector without ears what it carries.
func openMMProj26B(tb testing.TB) *tensors.GGUF {
	tb.Helper()
	g, err := tensors.OpenGGUF(mmproj26BPath(tb))
	if err != nil {
		tb.Fatalf("the 26B projector will not open: %v", err)
	}
	tb.Cleanup(func() { g.Close() })
	return g
}

func TestAudioConfigReadsTheProjector(t *testing.T) {
	g := openMMProj(t)
	cfg, err := LoadAudioConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Unified {
		t.Fatal("GOLEM_MMPROJ is E2B's, which carries a conformer")
	}
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"blocks", cfg.Blocks, 12},
		{"width", cfg.Dim, 1024},
		{"feed forward", cfg.FFN, 4096},
		{"heads", cfg.Heads, 8},
		{"head width", cfg.HeadDim, 128},
		{"mel bins", cfg.MelBins, 128},
		{"projection", cfg.ProjDim, 1536},
		{"chunk", cfg.Chunk, 12},
		{"past horizon", cfg.Past, 12},
		{"context", cfg.Context, 24},
		{"relative positions", cfg.RPE, 13},
	} {
		if c.got != c.want {
			t.Errorf("%s: %d, want %d", c.name, c.got, c.want)
		}
	}
	if cfg.Eps != 1e-6 {
		t.Errorf("epsilon is %g, want 1e-6", cfg.Eps)
	}
	if cfg.Softcap != 50 {
		t.Errorf("the softcap is %g, want 50", cfg.Softcap)
	}
}

// The 12B's projector is the other kind: no encoder, forty milliseconds of
// waveform a token.
func TestAudioConfigReadsTheUnifiedProjector(t *testing.T) {
	if os.Getenv("GOLEM_MMPROJ_12B") == "" {
		t.Skip("set GOLEM_MMPROJ_12B to the 12B's projector to run this test")
	}
	cfg, err := LoadAudioConfig(openMMProj12(t))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Unified {
		t.Fatal("GOLEM_MMPROJ_12B declares a conformer, which the 12B has not")
	}
	if cfg.FrameSize != 640 {
		t.Errorf("a frame is %d samples, want 640", cfg.FrameSize)
	}
	if cfg.Blocks != 0 {
		t.Errorf("%d blocks on a projector with no encoder", cfg.Blocks)
	}
	if cfg.ProjDim != 3840 {
		t.Errorf("the projection produces %d-wide tokens, want 3840", cfg.ProjDim)
	}
}

// A projector with no audio encoder is not an error at load time; it is a
// model that cannot listen, and says so when asked to.
func TestAProjectorWithoutEarsSaysSo(t *testing.T) {
	g := openMMProj26B(t)
	if HasAudio(g) {
		t.Fatal("the 26B projector declares no audio encoder")
	}
	if _, err := LoadAudioConfig(g); err == nil {
		t.Fatal("LoadAudioConfig accepted a projector with no audio encoder")
	}
}

// And one that does say so, says so.
func TestTheE2BProjectorDeclaresEars(t *testing.T) {
	if !HasAudio(openMMProj(t)) {
		t.Fatal("the E2B projector carries a conformer and did not declare it")
	}
}
