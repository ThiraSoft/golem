package pockettts

import (
	"os"
	"path/filepath"
	"testing"
)

// The reference recording of the gladyss voice, and the state the Python daemon
// encoded from it. Both live outside this repository — they belong to whoever
// recorded and encoded them — so the test skips when they are not there.
const (
	refWAV   = "/home/ricardo/dev/gladyss/voix/gladyss.wav"
	refState = "/home/ricardo/dev/gladyss/voix/gladyss.safetensors"
)

// TestVoiceFromWAVMatchesTheDaemon encodes a recording and compares the state
// it produces against the one PyTorch produced from the same file.
//
// This is the only test that can say the cloning is right: every stage before
// it is checked against a fixture, but a voice is what comes out of all of them
// at once, and a voice that is subtly wrong sounds like a different person
// rather than like an error.
func TestVoiceFromWAVMatchesTheDaemon(t *testing.T) {
	if _, err := os.Stat(refWAV); err != nil {
		t.Skip("no reference recording")
	}
	engine, _ := testEngine(t)

	got, err := engine.VoiceFromWAV(refWAV)
	if err != nil {
		t.Fatal(err)
	}
	want, err := engine.LoadVoice(refState)
	if err != nil {
		t.Skipf("no reference state: %v", err)
	}

	if got.position != want.position {
		t.Fatalf("voice of %d positions, the daemon wrote %d", got.position, want.position)
	}

	var worst float32
	for i := range got.state.Caches {
		g, w := got.state.Caches[i], want.state.Caches[i]
		n := got.position * engine.trans.Config.NumHeads * (engine.trans.Config.DModel / engine.trans.Config.NumHeads)
		for j := 0; j < n; j++ {
			for _, d := range []float32{g.K[j] - w.K[j], g.V[j] - w.V[j]} {
				if d < 0 {
					d = -d
				}
				if d > worst {
					worst = d
				}
			}
		}
	}
	t.Logf("largest gap in the caches: %g", worst)
	if worst > 5e-3 {
		t.Errorf("caches differ by %g", worst)
	}
}

// A voice saved and read back is the same voice.
func TestSaveVoiceRoundTrip(t *testing.T) {
	if _, err := os.Stat(refWAV); err != nil {
		t.Skip("no reference recording")
	}
	engine, _ := testEngine(t)

	v, err := engine.VoiceFromWAV(refWAV)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "voice.safetensors")
	if err := engine.SaveVoice(path, v); err != nil {
		t.Fatal(err)
	}
	back, err := engine.LoadVoice(path)
	if err != nil {
		t.Fatal(err)
	}
	if back.position != v.position {
		t.Fatalf("read back %d positions, wrote %d", back.position, v.position)
	}
	for i := range v.state.Caches {
		a, b := v.state.Caches[i], back.state.Caches[i]
		n := v.position * engine.trans.Config.NumHeads * (engine.trans.Config.DModel / engine.trans.Config.NumHeads)
		for j := 0; j < n; j++ {
			if a.K[j] != b.K[j] || a.V[j] != b.V[j] {
				t.Fatalf("layer %d, position %d: the round trip changed the cache", i, j)
			}
		}
	}
}
