package qwen

import "testing"

// The sixteen tokens llama.cpp drew greedily after the prompt, replayed one at
// a time.
//
// Sixteen identical identifiers is a stronger assertion than any of the
// activation comparisons: it says the head, the cache and the position
// arithmetic are all right together, not merely close. A divergence at step 0
// is the head; at step 1 it is the move from a batched prompt to a single
// position; later than that it is the cache.
func TestGreedyContinuationMatchesReference(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openModel(t)

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
