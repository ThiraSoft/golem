package gemma

import "testing"

// A batch must be the same computation as the tokens one at a time, bit for
// bit. Nothing about the arithmetic changes when several positions share a
// pass — only the order in which the weights are read — so anything less than
// equality is a bug in the batching, not a rounding.
func TestBatchAgreesWithOneAtATime(t *testing.T) {
	tokens := []int32{2, 105, 1596, 236764, 508, 11065, 3, 42, 7, 999}

	one := openEngine(t, 4096)
	want := make([][]float32, len(tokens))
	for i, token := range tokens {
		want[i] = append([]float32(nil), one.Forward(token, i)...)
	}

	many := openEngine(t, 4096)
	got := many.ForwardBatch(tokens, 0)

	for i := range tokens {
		for j := range got[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("position %d, element %d: %v against %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

// And a batch that starts part way through a conversation must see what came
// before it: the cache is the only thing carrying the earlier positions.
func TestBatchAfterAPrompt(t *testing.T) {
	prompt := []int32{2, 105, 1596, 236764}
	rest := []int32{508, 11065, 3, 42}

	one := openEngine(t, 4096)
	for i, token := range prompt {
		one.Forward(token, i)
	}
	want := make([][]float32, len(rest))
	for i, token := range rest {
		want[i] = append([]float32(nil), one.Forward(token, len(prompt)+i)...)
	}

	many := openEngine(t, 4096)
	many.ForwardBatch(prompt, 0)
	got := many.ForwardBatch(rest, len(prompt))

	for i := range rest {
		for j := range got[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("position %d, element %d: %v against %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}
