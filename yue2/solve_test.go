package yue2

import (
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// TestNARSolve tests the midpoint flow matching solver over the 100-frame short
// replay with 32 steps. It checks waypoints xt_0, xt_1, xt_7, xt_31 and the final
// latent, asserting a signal-to-noise ratio of at least 30 dB on the latent.
func TestNARSolve(t *testing.T) {
	n, _ := sharedNAR(t)
	fix := loadFixtures(t)
	noise := fix.read(t, "nar/noise")

	steps := 32
	frames := narFrames

	recorded := make(map[string][]float32)
	hook := func(step int, state []float32) {
		st := make([]float32, len(state))
		copy(st, state)
		switch step {
		case 0:
			recorded["xt_0"] = st
		case 1:
			recorded["xt_1"] = st
		case 7:
			recorded["xt_7"] = st
		case 31:
			recorded["xt_31"] = st
		}
	}

	gotLatent, err := n.solveWithHook(noise, frames, steps, nil, hook)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}

	checkWaypoint := func(name, fixName string, got []float32, minSNR float64) {
		t.Helper()
		want := fix.read(t, fixName)
		if len(got) != len(want) {
			t.Fatalf("%s length mismatch: got %d, want %d", name, len(got), len(want))
		}
		rms := rmsError(got, want)
		snr := snrDB(got, want)
		t.Logf("%s: rms error %.4f%%, SNR %.2f dB", name, rms*100, snr)
		if minSNR > 0 && snr < minSNR {
			t.Errorf("%s: SNR %.2f dB below bar %.2f dB (rms error %.4f%%)", name, snr, minSNR, rms*100)
		}
	}

	checkWaypoint("xt_0", "nar/xt_0", recorded["xt_0"], 0)
	checkWaypoint("xt_1", "nar/xt_1", recorded["xt_1"], 0)
	checkWaypoint("xt_7", "nar/xt_7", recorded["xt_7"], 0)
	checkWaypoint("xt_31", "nar/xt_31", recorded["xt_31"], 0)
	checkWaypoint("latent", "nar/latent", gotLatent, 30.0)
}

func BenchmarkNARStep(b *testing.B) {
	heavy.Skip(b, "a measurement over 4870 frames")
	const frames, arLen = 4870, 2600
	bb := loadNARWeights(b)
	d, err := vk.Open()
	if err != nil {
		b.Skipf("no Vulkan device: %v", err)
	}
	defer d.Close()
	if !d.Coopmat() {
		b.Skip("the NAR wants cooperative matrices")
	}
	n, err := NewNAR(d, bb, frames, arLen)
	bb.Close()
	if err != nil {
		b.Fatal(err)
	}
	defer n.Close()
	rng := rand.New(rand.NewSource(1))
	width := n.cfg.KVHeads * n.cfg.HeadDim
	k, v := make([][]uint16, n.cfg.Layers), make([][]uint16, n.cfg.Layers)
	for l := range k {
		k[l], v[l] = make([]uint16, arLen*width), make([]uint16, arLen*width)
		for i := range k[l] {
			k[l][i] = nn.Half(float32(rng.NormFloat64()))
			v[l][i] = nn.Half(float32(rng.NormFloat64()))
		}
	}
	if err := n.SetPrefix(k, v, arLen); err != nil {
		b.Fatal(err)
	}
	x := make([]float32, 64*frames)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	b.Logf("weights %.2f GB, cache %.2f GB, arena %.2f GB",
		float64(n.WeightBytes())/1e9, float64(n.CacheBytes())/1e9, float64(n.ArenaBytes())/1e9)

	// Warm up one step (records programs and shape tables)
	if _, err := n.Solve(x, frames, 1, nil); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := n.Solve(x, frames, 1, nil); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N), "ms/step")
}
