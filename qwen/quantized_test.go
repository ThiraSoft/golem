package qwen

// The same engine on weights that went through four bits.
//
// The BF16 checkpoint proved the architecture: whatever the arithmetic did, it
// was not the quantizer's doing. This file proves the other half — that the
// Q4_0 kernels are right for these shapes — by running the same comparisons
// against a recording made from the quantized file. Its waypoints are not the
// BF16 ones and are not meant to be: they are what llama.cpp computes from the
// same weights this engine reads.
//
// The file is made with `llama-quantize --pure`, because the Q4_0 build
// published alongside the BF16 one stores ffn_down as Q4_1, which
// tensors/gguf.go refuses and should go on refusing.
//
// GOLEM_MODEL_QWEN_Q4 names it; a machine without it skips.

import (
	"os"
	"testing"
)

func quantizedPath(tb testing.TB) string {
	tb.Helper()
	path := os.Getenv("GOLEM_MODEL_QWEN_Q4")
	if path == "" {
		tb.Skip("set GOLEM_MODEL_QWEN_Q4 to a pure Q4_0 Qwen3 GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("GOLEM_MODEL_QWEN_Q4 names %s, which is not there", path)
	}
	return path
}

func openQuantized(t *testing.T) *Model {
	t.Helper()
	m, err := Open(quantizedPath(t), 4096)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// Every tensor of the quantized file is a format this engine reads. A K-quant
// or a Q4_1 left behind by --pure would fail here rather than three tests
// later.
func TestQuantizedWeightsLoad(t *testing.T) {
	m := openQuantized(t)
	if m.W.ActBF16 {
		t.Error("a Q4_0 checkpoint should not ask for bfloat16 activations")
	}
	if q := m.W.Blocks[0].Q.Quant.String(); q != "Q4_0" {
		t.Errorf("blk.0.attn_q is %s, so this file is not pure Q4_0", q)
	}
}

// The tolerances here are wider than the bfloat16 run's, and the reason is the
// quantizer rather than the engine. Every product on this path quantizes its
// activation to Q8_0 in blocks of thirty-two, and a value sitting near a step
// rounds the other way from llama.cpp's now and then; twenty-eight blocks
// accumulate that. The observed worst is 2.7e-2 at block 27, against 4.2e-3
// for the same architecture in bfloat16.
//
// That path is chaotic rather than biased, which is worth knowing before
// reading a change in these numbers as a regression. Computing the feed
// forward's activation with the exact exponential instead of ggml's polynomial
// — a difference of about two units in the last place per element — moves this
// figure between 1.1e-2 and 2.7e-2, while leaving ffn_out itself agreeing with
// the reference to exactly the same seven digits either way. Neither is closer
// to llama.cpp; the buckets simply fall differently.
//
// So what says this is drift and not a mistake is not the tolerance, and not
// this number moving. It is the two assertions below that do not move at all:
// the top sixty-four logits land within 0.26, and all sixteen greedy tokens are
// the ones llama.cpp drew.
func TestQuantizedFullStackMatchesReference(t *testing.T) {
	f := loadFixture(t, "layers_q4")
	m := openQuantized(t)

	hidden := m.ForwardBatch(f.Tokens, 0)
	last := len(f.Tokens) - 1

	for i := 0; i < f.NLayer; i++ {
		name := "l_out-" + itoa(i)
		compareRelative(t, name+" at the last position", m.BlockOutput(i), f.lastColumn(t, name), 3.5e-2)
	}
	compareRelative(t, "result_norm", hidden[last], f.lastColumn(t, "result_norm"), 5e-2)
}

func TestQuantizedLogitsMatchReference(t *testing.T) {
	f := loadFixture(t, "layers_q4")
	m := openQuantized(t)

	hidden := m.ForwardBatch(f.Tokens, 0)
	out := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden[len(hidden)-1], out)

	got := make([]float32, len(f.LogitsTop))
	want := make([]float32, len(f.LogitsTop))
	for i, probe := range f.LogitsTop {
		got[i], want[i] = out[probe.ID], probe.Logit
	}
	compare(t, "the top 64 logits", got, want, 0.5)

	if best := argmax(out); best != f.Argmax {
		t.Errorf("argmax %d, want %d", best, f.Argmax)
	}
}

// The sixteen greedy tokens again, from four-bit weights. Quantization moves
// every activation a little; if it moved the drawn token, the engine and its
// reference would be disagreeing about the model rather than about a digit.
func TestQuantizedGreedyContinuationMatchesReference(t *testing.T) {
	f := loadFixture(t, "layers_q4")
	m := openQuantized(t)

	hidden := m.ForwardBatch(f.Tokens, 0)
	out := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden[len(hidden)-1], out)
	next := argmax(out)

	for step, want := range f.Greedy {
		if next != want {
			t.Fatalf("step %d: drew %d, want %d", step, next, want)
		}
		hidden = m.ForwardBatch([]int32{next}, len(f.Tokens)+step)
		m.Logits(hidden[0], out)
		next = argmax(out)
	}
}
