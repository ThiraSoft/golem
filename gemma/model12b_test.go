package gemma

// The 12B, which declares the same architecture as E2B and disagrees with it
// about almost everything measurable: no per-layer embeddings, sixteen query
// heads over eight key-value ones in a window block and one in a global block,
// and a global block that publishes no value projection at all — its keys are
// its values.
//
// Its fixtures are a second recording by the same tool, in testdata/gemma/
// layers12, and the checkpoint they were made from is named by GOLEM_MODEL_12B.
// A machine that has only one of the two models skips the other's tests.

import (
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

func model12BPath(tb testing.TB) string {
	tb.Helper()
	path := os.Getenv("GOLEM_MODEL_12B")
	if path == "" {
		tb.Skip("set GOLEM_MODEL_12B to a Gemma 4 12B GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("GOLEM_MODEL_12B names %s, which is not there", path)
	}
	return path
}

func open12B(t *testing.T) *tensors.GGUF {
	t.Helper()
	g, err := tensors.OpenGGUF(model12BPath(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	return g
}

func open12BEngine(t *testing.T) *Model {
	t.Helper()
	m, err := Open(model12BPath(t), 4096)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func load12B(t *testing.T) (*fixture, *Config, *Weights) {
	t.Helper()
	f := loadFixture(t, "layers12")
	g := open12B(t)
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return f, cfg, w
}

// What the file says it is, against what it is known to be.
func TestLoadConfig12B(t *testing.T) {
	g := open12B(t)
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Dim != 3840 || len(cfg.Blocks) != 48 {
		t.Fatalf("%d blocks of %d, expected 48 of 3840", len(cfg.Blocks), cfg.Dim)
	}
	if cfg.PLEDim != 0 {
		t.Fatalf("the 12B has no per-layer embeddings, and the file declares %d", cfg.PLEDim)
	}
	if len(cfg.Suppress) == 0 {
		t.Fatal("the file forbids the image and audio markers, and none was read")
	}
	if !cfg.EmptyThought {
		t.Fatal("the 12B's template closes a generation prompt with an empty thought channel")
	}
	// Every sixth block is global, owns its cache, and has no value projection.
	for i, b := range cfg.Blocks {
		global := i%6 == 5
		if b.Window == global {
			t.Fatalf("block %d: window is %v", i, b.Window)
		}
		if !b.OwnsKV {
			t.Fatalf("block %d shares a cache, and no block of the 12B does", i)
		}
		if b.ValueIsKey != global {
			t.Fatalf("block %d: value from key is %v", i, b.ValueIsKey)
		}
		if b.Heads != 16 {
			t.Fatalf("block %d has %d query heads", i, b.Heads)
		}
	}
}

// Block 0 owns a window cache with eight key-value heads; block 5 is the first
// global one, the kind whose keys are also its values.
func TestBlock12B(t *testing.T) {
	f, cfg, w := load12B(t)
	for _, il := range []int{0, 5} {
		cache, s := NewCache(cfg), NewScratch(cfg)
		replayFullBlock(t, f, cfg, w, cache, s, il, true)
	}
}

// The whole stack on the reference's own prompt, block by block and then to the
// final norm.
//
// The tolerances are looser than E2B's, and the reason is the depth: the same
// Q4_0 rounding that leaves E2B 1.9e-2 from the reference after 35 blocks
// leaves the 12B 6.5e-2 after 48 — measured, not allowed for. Every block of it
// is within 2e-3 of the reference when driven by the reference's own input,
// which is what TestBlock12B holds, and the logits still choose the same
// tokens.
func TestForward12BMatchesTheReference(t *testing.T) {
	f := loadFixture(t, "layers12")
	m := open12BEngine(t)

	for pos, token := range f.Tokens {
		m.Forward(token, pos)
		for _, il := range []int{0, 5, 24, 47} {
			compareRelative(t, "l_out-"+itoa(il)+" at position "+itoa(pos),
				m.BlockOutput(il), f.column(t, "l_out-"+itoa(il), pos), 8e-2)
		}
	}
	m.Reset()
	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}
	compareRelative(t, "result_norm", hidden, f.tensor(t, "result_norm"), 7e-2)
}

// The logits, and the token the reference would have chosen from them.
func TestLogits12BMatchTheReference(t *testing.T) {
	f := loadFixture(t, "layers12")
	m := open12BEngine(t)

	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}

	ids := make([]int32, len(f.LogitsTop))
	want := make([]float32, len(f.LogitsTop))
	for i, probe := range f.LogitsTop {
		ids[i], want[i] = probe.ID, probe.Logit
	}
	// A logit of the 12B is further from the reference's than E2B's is — 0.86
	// measured here against 0.33 there, on values around fifteen. Thirteen more
	// blocks of Q4_0 products accumulate that much more rounding, and every
	// waypoint above is inside the tolerances E2B holds to.
	got := make([]float32, len(ids))
	m.LogitsAt(hidden, ids, got)
	compare(t, "the top 64 logits", got, want, 1.0)

	all := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden, all)
	if best := Argmax(all); best != f.Argmax {
		t.Fatalf("the model would choose %d, the reference chose %d", best, f.Argmax)
	}
	// The tokens the file forbids are forbidden, whichever way the logits fell.
	for _, id := range m.Cfg.Suppress {
		if !math.IsInf(float64(all[id]), -1) {
			t.Fatalf("token %d is suppressed by the file and scores %v", id, all[id])
		}
	}
}

// The reference's own continuation, token for token. The prompt is not a chat
// turn, so both engines run into the same degenerate repetition; what is tested
// is that they run into it together.
func TestGreedy12BMatchesTheReference(t *testing.T) {
	f := loadFixture(t, "layers12")
	m := open12BEngine(t)

	pos := 0
	var hidden []float32
	for _, token := range f.Tokens {
		hidden = m.Forward(token, pos)
		pos++
	}

	// The same rule as E2B's: a choice inside the gap the logits are known to
	// carry is a tie, not a fault, and the reference's token is fed back.
	const tie = 1.0
	logits := make([]float32, m.Cfg.Vocab)
	for step, want := range f.Greedy {
		m.Logits(hidden, logits)
		if got := Argmax(logits); got != want {
			if margin := logits[got] - logits[want]; margin > tie {
				t.Fatalf("step %d: chose %d over the reference's %d by %v, which is past a tie",
					step, got, want, margin)
			} else {
				t.Logf("step %d: chose %d over %d by %v, a tie inside the measured gap",
					step, got, want, margin)
			}
		}
		hidden = m.Forward(want, pos)
		pos++
	}
}
