package main

import (
	"context"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/engine"
)

// Scoring the last positions of a prompt rather than the last alone: the
// distribution before each of them comes back, in order.
func TestPrefillScoredScoresTheLastPositions(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	outs := [][]float32{scores(), scores()}
	if _, err := c.PrefillScored(context.Background(), engine.TextPrompt([]int32{1, 2, 3, 4}), outs); err != nil {
		t.Fatal(err)
	}
	if outs[0][0] != 3 || outs[1][0] != 4 {
		t.Fatalf("scored %v and %v, want the states of 3 and 4", outs[0][0], outs[1][0])
	}
}

// The positions scored are fed even when the cache already holds them: a
// state that was not kept cannot be scored.
func TestPrefillScoredFeedsWhatItScores(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	if _, err := c.Prefill([]int32{1, 2, 3, 4}, scores()); err != nil {
		t.Fatal(err)
	}
	e.fed = nil
	outs := [][]float32{scores(), scores(), scores()}
	if _, err := c.PrefillScored(context.Background(), engine.TextPrompt([]int32{1, 2, 3, 4}), outs); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(e.fed, []int32{2, 3, 4}) {
		t.Fatalf("fed %v, want 2 3 4", e.fed)
	}
}

// A prompt longer than a pass is cut so that every position scored is in the
// last pass, which is the only one whose states are scored.
func TestPrefillScoredKeepsTheScoredPositionsInOnePass(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	ids := make([]int32, promptBatch+1)
	for i := range ids {
		ids[i] = int32(i + 1)
	}
	outs := [][]float32{scores(), scores()}
	if _, err := c.PrefillScored(context.Background(), engine.TextPrompt(ids), outs); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(e.widths, []int{promptBatch - 1, 2}) {
		t.Fatalf("passes of %v", e.widths)
	}
	if outs[0][0] != float32(promptBatch) || outs[1][0] != float32(promptBatch+1) {
		t.Fatalf("scored %v and %v", outs[0][0], outs[1][0])
	}
}

// An engine whose next-token distribution is a table keyed by the token just
// fed: what follows token k is next[k], as logits. Everything absent is -inf
// but for a floor, so a log-softmax is finite everywhere.
type tableEngine struct {
	recordingEngine
	next map[int32][]float32
}

func (e *tableEngine) LogitsBatch(hidden [][]float32, out [][]float32) {
	for i, o := range out {
		for j := range o {
			o[j] = -30
		}
		copy(o, e.next[int32(hidden[i][0])])
	}
}

func (e *tableEngine) Logits(hidden []float32, out []float32) {
	e.LogitsBatch([][]float32{hidden}, [][]float32{out})
}

func logSoftmaxAt(row []float32, id int) float64 {
	m := math.Inf(-1)
	for _, v := range row {
		m = math.Max(m, float64(v))
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v) - m)
	}
	return float64(row[id]) - m - math.Log(sum)
}

// An option's weight is the sum of the log-probabilities of its tokens, each
// read off the state before it; options that share a beginning share its
// passes.
func TestWeighSumsEachOptionsTokens(t *testing.T) {
	// The prompt is 1 2, a question's own words 3, then options:
	// 5, 6 7 and 6 8.
	row := func(vals map[int]float32) []float32 {
		r := make([]float32, 10)
		for i := range r {
			r[i] = -30
		}
		for k, v := range vals {
			r[k] = v
		}
		return r
	}
	e := &tableEngine{next: map[int32][]float32{
		3: row(map[int]float32{5: 1, 6: 2}),
		6: row(map[int]float32{7: 0.5, 8: -0.5}),
	}}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	text := func(ids []int32) (engine.Prompt, error) { return engine.TextPrompt(ids), nil }
	got, _, err := c.Weigh(context.Background(), text, []int32{1, 2},
		[]weighing{{tail: []int32{3}, options: [][]int32{{5}, {6, 7}, {6, 8}}}}, 10)
	if err != nil {
		t.Fatal(err)
	}
	r3, r6 := e.next[3], e.next[6]
	want := []float64{
		logSoftmaxAt(r3, 5),
		logSoftmaxAt(r3, 6) + logSoftmaxAt(r6, 7),
		logSoftmaxAt(r3, 6) + logSoftmaxAt(r6, 8),
	}
	for i := range want {
		if math.Abs(got[0][i]-want[i]) > 1e-9 {
			t.Fatalf("option %d weighs %v, want %v", i, got[0][i], want[i])
		}
	}
}

// Two questions about one state read the state once: the second one's passes
// start where the state ends.
func TestWeighReadsTheStateOnce(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	text := func(ids []int32) (engine.Prompt, error) { return engine.TextPrompt(ids), nil }
	state := []int32{1, 2, 3, 4, 5, 6}
	_, _, err := c.Weigh(context.Background(), text, state, []weighing{
		{tail: []int32{7}, options: [][]int32{{1}, {2}}},
		{tail: []int32{8, 9}, options: [][]int32{{1}, {2, 3}}},
	}, 8)
	if err != nil {
		t.Fatal(err)
	}
	// The state and the first question; the second question; and the first
	// token of its second option, whose second token needs the state after it.
	want := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 2}
	if !slices.Equal(e.fed, want) {
		t.Fatalf("fed %v, want %v", e.fed, want)
	}
	for i, p := range e.posOf[6:] {
		if wantPos := []int{6, 6, 7, 8}[i]; p != wantPos {
			t.Fatalf("positions %v after the state", e.posOf[6:])
		}
	}
}
