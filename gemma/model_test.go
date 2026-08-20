package gemma

import (
	"math"
	"testing"
)

func openEngine(tb testing.TB, maxContext int) *Model {
	tb.Helper()
	m, err := Open(modelPath(tb), maxContext)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { m.Close() })
	return m
}

// Thirty-five blocks on the reference's own prompt. Only the last position is
// recorded, because that is the one whose logits llama.cpp was asked for.
func TestForwardMatchesTheReference(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openEngine(t, 4096)

	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}
	compareRelative(t, "result_norm", hidden, f.tensor(t, "result_norm"), 2e-2)
}

// Every block's output, so that a divergence names the block it started in
// rather than the model.
func TestForwardBlockByBlock(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openEngine(t, 4096)

	for pos, token := range f.Tokens {
		m.Forward(token, pos)
		for _, il := range []int{0, 4, 14, 15, 19, 34} {

			compareRelative(t, "l_out-"+itoa(il)+" at position "+itoa(pos),
				m.BlockOutput(il), f.column(t, "l_out-"+itoa(il), pos), 3e-2)
		}
	}
}

func TestLogitsMatchTheReference(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openEngine(t, 4096)

	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}

	ids := make([]int32, len(f.LogitsTop))
	want := make([]float32, len(f.LogitsTop))
	for i, probe := range f.LogitsTop {
		ids[i], want[i] = probe.ID, probe.Logit
	}
	got := make([]float32, len(ids))
	m.LogitsAt(hidden, ids, got)
	compare(t, "the top 64 logits", got, want, 0.5)

	// The whole vocabulary, and the token the reference would have chosen.
	all := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden, all)
	best := int32(0)
	for i, v := range all {
		if v > all[best] {
			best = int32(i)
		}
	}
	if best != f.Argmax {
		t.Fatalf("the model would choose %d, the reference chose %d", best, f.Argmax)
	}
	if d := math.Abs(float64(all[ids[0]] - want[0])); d > 0.5 {
		t.Fatalf("the whole-vocabulary logit disagrees with the sparse one by %g", d)
	}
	// Softcapping keeps everything inside the declared limit.
	for i, v := range all {
		if v <= -m.Cfg.LogitSoftcap || v >= m.Cfg.LogitSoftcap {
			t.Fatalf("logit %d is %v, outside the softcap of %v", i, v, m.Cfg.LogitSoftcap)
		}
	}
}

// Six hundred tokens, which is what makes the sliding window matter: below 512
// a window block and a global one see exactly the same positions, and a missing
// mask costs nothing. This is the test that fails when the window is wrong.
func TestWindowMasksBeyondItsReach(t *testing.T) {
	f := loadFixture(t, "window")
	m := openEngine(t, 4096)

	for pos, token := range f.Tokens {
		m.Forward(token, pos)
	}
	last := len(f.Tokens) - 1

	compareRelative(t, "l_out-0 at position "+itoa(last), m.BlockOutput(0), f.tensor(t, "l_out-0"), 1e-2)
	compareRelative(t, "l_out-4 at position "+itoa(last), m.BlockOutput(4), f.tensor(t, "l_out-4"), 1e-2)
	compareRelative(t, "l_out-15 at position "+itoa(last), m.BlockOutput(15), f.tensor(t, "l_out-15"), 1e-2)
}
