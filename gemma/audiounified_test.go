package gemma

import (
	"math"
	"os"
	"sort"
	"testing"
)

// The 12B's projector holds one tensor and no block: the waveform in frames of
// 640, normalized and projected once. The recording is audio12.
func TestUnifiedAudioMatchesTheReference(t *testing.T) {
	if os.Getenv("GOLEM_MMPROJ_12B") == "" {
		t.Skip("set GOLEM_MMPROJ_12B to the 12B's projector to run this test")
	}
	f := loadAudioFixture(t, "audio12")
	tower := openAudioTower12(t)
	if !tower.Cfg.Unified {
		t.Fatal("GOLEM_MMPROJ_12B is not a gemma4ua projector")
	}
	rows := tower.Encode(loadSpeech(t))
	if len(rows) != f.NAudioTokens {
		t.Fatalf("%d soft tokens, the recording has %d", len(rows), f.NAudioTokens)
	}
	want := f.tensor(t, "projected")
	got := make([]float32, 0, len(want))
	for _, r := range rows {
		got = append(got, r...)
	}
	// There is nothing between the waveform and these rows but a norm and one
	// product, so the agreement here is as tight as bfloat16 allows and a
	// single bound is enough.
	closeRelative(t, "projected", got, want, 1e-3)
}

// Forty milliseconds a token, and the last frame padded with silence rather
// than dropped.
func TestUnifiedFramesAreSixHundredAndForty(t *testing.T) {
	if os.Getenv("GOLEM_MMPROJ_12B") == "" {
		t.Skip("set GOLEM_MMPROJ_12B to the 12B's projector to run this test")
	}
	tower := openAudioTower12(t)
	rows := tower.Encode(make([]float32, 640*3+1))
	if len(rows) != 4 {
		t.Fatalf("%d tokens for 1921 samples, want 4", len(rows))
	}
	if len(rows[0]) != tower.Cfg.ProjDim {
		t.Fatalf("a row is %d wide, want %d", len(rows[0]), tower.Cfg.ProjDim)
	}
}

// The norm is over a frame's own 640 samples and nothing else: two recordings
// that differ by a constant gain give the same rows, to within what the
// epsilon inside the norm allows on a quiet frame.
func TestUnifiedNormalizesEachFrameOnItsOwn(t *testing.T) {
	if os.Getenv("GOLEM_MMPROJ_12B") == "" {
		t.Skip("set GOLEM_MMPROJ_12B to the 12B's projector to run this test")
	}
	tower := openAudioTower12(t)
	quiet := loadSpeech(t)[:640*20]
	loud := make([]float32, len(quiet))
	for i, v := range quiet {
		loud[i] = v * 4
	}
	a, b := tower.Encode(quiet), tower.Encode(loud)
	gaps := make([]float64, 0, len(a)*len(a[0]))
	var scale float32
	for i := range a {
		for j := range a[i] {
			gaps = append(gaps, math.Abs(float64(a[i][j]-b[i][j])))
			if v := abs32(a[i][j]); v > scale {
				scale = v
			}
		}
	}
	sort.Float64s(gaps)
	if worst, allowed := gaps[len(gaps)-1], float64(scale)*1e-2; worst > allowed {
		t.Fatalf("four times the gain moved a row by %g, over the %g allowed on a range of %g; the norm is not per frame",
			worst, allowed, scale)
	}
}
