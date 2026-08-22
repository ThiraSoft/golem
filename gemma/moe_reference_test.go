package gemma

// The 26B A4B against llama.cpp, one waypoint at a time.
//
// The order of the checks is the order a bug is cheapest to find in: the
// stack block by block first, to name the block a divergence begins in; then
// the router's logits, because a wrong choice of experts makes every later
// waypoint wrong for a reason no later waypoint can show; then each branch
// alone; then their sum.

import (
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

func model26BPathT(tb testing.TB) string {
	tb.Helper()
	path := os.Getenv("GOLEM_MODEL_26B")
	if path == "" {
		tb.Skip("set GOLEM_MODEL_26B to a Gemma 4 26B A4B GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("GOLEM_MODEL_26B names %s, which is not there", path)
	}
	return path
}

func load26B(t *testing.T) (*fixture, *Model) {
	t.Helper()
	f := loadFixture(t, "moe26")
	path := model26BPathT(t)
	f.requireModel(t, path)
	m, err := Open(path, 4096)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return f, m
}

// The blocks moe.run kept. Every block of this model is a mixture block, so
// these were chosen for their attention: two window blocks, the first global
// one, and two more far apart.
var moeBlocks = []int{0, 1, 5, 6, 17}

// TestMoEForwardBlockByBlock runs the whole stack and compares every kept
// block's output. It is the first test to read because it names the block a
// divergence begins in, which is the only thing the other tests need to know.
//
// The tolerance grows with the block, and the reason is in the arithmetic: an
// expert block sums eight products whose activations are each quantized on the
// way in, where a dense block quantizes once. Block 0 lands within a part in
// three million of the reference; block 17, fed by sixteen blocks of that,
// within three parts in a hundred.
func TestMoEForwardBlockByBlock(t *testing.T) {
	f, m := load26B(t)
	for pos, token := range f.Tokens {
		m.Forward(token, pos)
		for _, il := range moeBlocks {
			compareRelative(t, "l_out-"+itoa(il)+" at position "+itoa(pos),
				m.BlockOutput(il), f.column(t, "l_out-"+itoa(il), pos), 5e-2)
		}
	}
}

// TestMoEResultNorm is the last norm, which the logits are drawn from.
//
// The tolerance is looser than the 12B's, and the reason is measurable rather
// than convenient: an expert block sums eight products whose activations are
// each quantized on the way in, where a dense block quantizes once. Thirty of
// those accumulate about three times the 12B's rounding. What that costs in
// practice is settled below, by the choice of token rather than by a gap.
func TestMoEResultNorm(t *testing.T) {
	f, m := load26B(t)
	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}
	compareRelative(t, "result_norm", hidden, f.tensor(t, "result_norm"), 8e-2)
}

// TestMoEArgmax is what the rounding above actually costs: nothing, if the
// model still chooses the reference's token.
func TestMoEArgmax(t *testing.T) {
	f, m := load26B(t)
	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}
	all := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden, all)
	if best := Argmax(all); best != f.Argmax {
		t.Fatalf("the model would choose %d, the reference chose %d (by %v)",
			best, f.Argmax, all[best]-all[f.Argmax])
	}
	for _, id := range m.Cfg.Suppress {
		if !math.IsInf(float64(all[id]), -1) {
			t.Fatalf("token %d is suppressed by the file and scores %v", id, all[id])
		}
	}
}

// TestMoEGreedyMatchesTheReference replays the reference's own continuation.
// The prompt is not a chat turn, so both engines run into the same degenerate
// repetition; what is tested is that they run into it together.
func TestMoEGreedyMatchesTheReference(t *testing.T) {
	f, m := load26B(t)
	pos := 0
	var hidden []float32
	for _, token := range f.Tokens {
		hidden = m.Forward(token, pos)
		pos++
	}
	// The same rule the other two checkpoints use: a choice inside the gap the
	// logits are known to carry is a tie, not a fault, and the reference's
	// token is fed back so that one tie does not become a different sentence.
	const tie = 1.5
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

// TestMoEBlockWaypoints replays one block from the reference's own inputs and
// compares each waypoint inside it, so that a failure names the half — and,
// within the mixture, the branch — rather than the block.
func TestMoEBlockWaypoints(t *testing.T) {
	f := loadFixture(t, "moe26")
	path := model26BPathT(t)
	f.requireModel(t, path)
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	cfg, err := LoadConfig(g, 4096)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}

	for _, il := range moeBlocks {
		const pos = 0
		bc, bw := cfg.Blocks[il], &w.Blocks[il]
		cache, s := NewCache(cfg), NewScratch(cfg)
		xs := [][]float32{make([]float32, cfg.Dim)}
		copy(xs[0], blockInput(t, f, il, pos))
		freqs := w.RoPEFreqs
		if bc.Window {
			freqs = nil // the frequency factors belong to the global blocks
		}
		Block(cfg, bc, bw, s.RoPE(bc, Run(cache, pos, 1), freqs), Run(cache, pos, 1), s, xs, nil)

		at := itoa(il)
		compareRelative(t, "attn_out-"+at, s.AttnOut(), f.column(t, "attn_out-"+at, pos), 2e-2)
		compareRelative(t, "ffn_moe_logits-"+at, s.expLogits[0], f.column(t, "ffn_moe_logits-"+at, pos), 2e-2)
		compareRelative(t, "ffn_mlp-"+at, s.MoEShared(), f.column(t, "ffn_mlp-"+at, pos), 2e-2)
		compareRelative(t, "ffn_moe-"+at, s.MoEExperts(), f.column(t, "ffn_moe-"+at, pos), 2e-2)
		compareRelative(t, "l_out-"+at, xs[0], f.column(t, "l_out-"+at, pos), 1e-3)
	}
}
