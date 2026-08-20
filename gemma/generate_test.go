package gemma

import "testing"

// The reference's own continuation, token for token. Greedy decoding, so there
// is nothing random to account for: the same weights and the same arithmetic
// have to make the same sixteen choices.
func TestGreedyGenerationMatchesTheReference(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openEngine(t, 4096)

	pos := 0
	var hidden []float32
	for _, token := range f.Tokens {
		hidden = m.Forward(token, pos)
		pos++
	}

	// A choice may differ from the reference's when the two tokens are within
	// the gap the logits themselves are known to carry — TestLogitsMatchTheReference
	// measures a third of a logit, and this prompt ends in a run of control
	// tokens that the model scores within a tenth of each other. A tie decided
	// the other way is not a fault; a token chosen over one the reference
	// preferred by more than that gap is. When it happens the reference's own
	// token is fed back, so that the steps after it still test something.
	const tie = 0.5
	logits := make([]float32, m.Cfg.Vocab)
	for step, want := range f.Greedy {
		m.Logits(hidden, logits)
		got := Argmax(logits)
		if got != want {
			margin := logits[got] - logits[want]
			if margin > tie {
				t.Fatalf("step %d: chose %d over the reference's %d by %v, which is past a tie",
					step, got, want, margin)
			}
			t.Logf("step %d: chose %d over %d by %v, a tie inside the measured gap",
				step, got, want, margin)
		}
		hidden = m.Forward(want, pos)
		pos++
	}
}

// Reset has to put the model back where it started, or a second conversation
// would answer the first one.
func TestResetForgetsTheConversation(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openEngine(t, 4096)

	run := func() []float32 {
		var hidden []float32
		for pos, token := range f.Tokens {
			hidden = m.Forward(token, pos)
		}
		return append([]float32(nil), hidden...)
	}

	first := run()
	m.Reset()
	second := run()
	compare(t, "the hidden state after a reset", second, first, 0)
}
