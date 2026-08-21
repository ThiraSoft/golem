package gemma

import (
	"testing"
	"time"
)

// What a pass costs as more conversations share it, and what the head costs
// beside it. One read of the weights serving n tokens is the whole claim of a
// mixed batch; this says by how much, and it is where cmd/golem-server's
// figures come from.
func TestMixedBatchCost(t *testing.T) {
	m := openEngine(t, 4096)
	const depth = 64
	for _, n := range []int{1, 2, 4, 8} {
		if err := m.SetSlots(n); err != nil {
			t.Fatal(err)
		}
		tokens := make([]int32, n)
		at := make([]Place, n)
		for i := range at {
			m.UseSlot(i)
			m.ForwardBatch(make([]int32, depth), 0)
			tokens[i] = 42
			at[i] = Place{Cache: m.Slot(i), Pos: depth}
		}
		m.ForwardMixed(tokens, at)
		start := time.Now()
		const reps = 5
		for r := 0; r < reps; r++ {
			m.ForwardMixed(tokens, at)
		}
		each := time.Since(start) / reps
		hidden := make([][]float32, n)
		out := make([][]float32, n)
		for i := range hidden {
			hidden[i] = make([]float32, m.Cfg.Dim)
			out[i] = make([]float32, m.Cfg.Vocab)
		}
		start = time.Now()
		for r := 0; r < reps; r++ {
			m.LogitsBatch(hidden, out)
		}
		head := time.Since(start) / reps
		t.Logf("%d conversations: pass %s, head %s, %.1f tokens/s",
			n, each.Round(time.Millisecond), head.Round(time.Millisecond),
			float64(n)/(each+head).Seconds())
	}
}

// Reading four prompts at once against reading one. A pass is thirty-two
// positions wide whichever way they are made up, so the question is whether
// four conversations of eight cost what one run of thirty-two costs.
func TestMixedPrefillCost(t *testing.T) {
	m := openEngine(t, 4096)
	const width = 32
	tokens := make([]int32, width)

	m.SetSlots(1)
	at := Run(m.Slot(0), 0, width)
	m.ForwardMixed(tokens, at)
	start := time.Now()
	for r := 0; r < 3; r++ {
		m.ForwardMixed(tokens, at)
	}
	one := time.Since(start) / 3

	if err := m.SetSlots(4); err != nil {
		t.Fatal(err)
	}
	four := make([]Place, 0, width)
	for i := 0; i < 4; i++ {
		four = append(four, Run(m.Slot(i), 0, width/4)...)
	}
	m.ForwardMixed(tokens, four)
	start = time.Now()
	for r := 0; r < 3; r++ {
		m.ForwardMixed(tokens, four)
	}
	split := time.Since(start) / 3

	t.Logf("32 positions of one conversation: %s (%.0f tokens/s)", one.Round(time.Millisecond), width/one.Seconds())
	t.Logf("8 positions of each of four:      %s (%.0f tokens/s)", split.Round(time.Millisecond), width/split.Seconds())
}
