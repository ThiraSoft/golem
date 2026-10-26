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

// branchEngine is a recordingEngine that can carry branches, and keeps what
// each token was told about its own.
type branchEngine struct {
	recordingEngine
	shifts []int
	holes  [][2][2]int
}

func (e *branchEngine) ForwardChunk(c engine.Chunk) [][]float32 {
	e.shifts = append(e.shifts, c.Shifts...)
	e.holes = append(e.holes, c.Holes...)
	return e.recordingEngine.chunk(c)
}

// chunk is what a recordingEngine makes of a pass that carries branches: the
// state of a token is the token, wherever it stands.
func (e *recordingEngine) chunk(c engine.Chunk) [][]float32 {
	e.slot = c.Slots[0]
	e.widths = append(e.widths, len(c.Tokens))
	hidden := make([][]float32, len(c.Tokens))
	for i, token := range c.Tokens {
		e.fed = append(e.fed, token)
		e.posOf = append(e.posOf, c.Positions[i])
		hidden[i] = []float32{float32(token)}
	}
	return hidden
}

func branching(t *testing.T, e interface {
	Engine
	engine.Branches
}) *Runner {
	r := running(t, e)
	r.SetBranches(e)
	return r
}

// Segments go after the prompt in its last pass, each after the one before
// it, turned back to where what it continues ends and blind to the rest. What
// is scored is the last positions each segment asks for.
func TestPrefillSegmentsLaysThemAfterThePrompt(t *testing.T) {
	e := &branchEngine{}
	c := NewContext(branching(t, e), 0, 4096, time.Now, 0)
	outs := [][]float32{scores(), scores(), scores(), scores()}
	segs := []segment{
		{tokens: []int32{4, 5}, parent: -1, scored: 1},
		{tokens: []int32{6}, parent: 0, scored: 1},
		{tokens: []int32{7, 8}, parent: 0, scored: 2},
	}
	if _, err := c.PrefillSegments(context.Background(), engine.TextPrompt([]int32{1, 2, 3}), segs, outs); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(e.fed, []int32{1, 2, 3, 4, 5, 6, 7, 8}) || !slices.Equal(e.widths, []int{8}) {
		t.Fatalf("fed %v in passes of %v", e.fed, e.widths)
	}
	if !slices.Equal(e.posOf, []int{0, 1, 2, 3, 4, 5, 6, 7}) || !slices.Equal(e.shifts, []int{0, 0, 0, 0, 0, 0, 1, 1}) {
		t.Fatalf("positions %v, shifted by %v", e.posOf, e.shifts)
	}
	none := [2][2]int{}
	if e.holes[3] != none || e.holes[5] != none || e.holes[6] != [2][2]int{{5, 6}} {
		t.Fatalf("holes %v", e.holes)
	}
	for i, want := range []float32{5, 6, 7, 8} {
		if outs[i][0] != want {
			t.Fatalf("scored %v at %d, want the state of %v", outs[i][0], i, want)
		}
	}
	// The cache holds the prompt: the segments are past it, for the next
	// prompt to write over.
	if c.Pos() != 3 {
		t.Fatalf("holds %d positions", c.Pos())
	}
}

// The second question's options see the state and their own question's
// words, which leaves a hole on each side of those words.
func TestPrefillSegmentsLeaveTwoHoles(t *testing.T) {
	e := &branchEngine{}
	c := NewContext(branching(t, e), 0, 4096, time.Now, 0)
	outs := [][]float32{scores(), scores(), scores(), scores(), scores()}
	segs := []segment{
		{tokens: []int32{4}, parent: -1, scored: 1},
		{tokens: []int32{5}, parent: 0, scored: 1},
		{tokens: []int32{6}, parent: -1, scored: 1},
		{tokens: []int32{7}, parent: 2, scored: 1},
		{tokens: []int32{8}, parent: 2, scored: 1},
	}
	if _, err := c.PrefillSegments(context.Background(), engine.TextPrompt([]int32{1, 2, 3}), segs, outs); err != nil {
		t.Fatal(err)
	}
	if got := e.shifts[3:]; !slices.Equal(got, []int{0, 0, 2, 2, 3}) {
		t.Fatalf("shifted by %v", got)
	}
	if e.holes[5] != [2][2]int{{3, 5}} || e.holes[7] != [2][2]int{{3, 5}, {6, 7}} {
		t.Fatalf("holes %v", e.holes[3:])
	}
}

// A second question about the same state feeds its own words from where the
// state ends, and the state's last position again, whose state was not kept.
func TestPrefillSegmentsRewindToTheState(t *testing.T) {
	e := &branchEngine{}
	c := NewContext(branching(t, e), 0, 4096, time.Now, 0)
	outs := [][]float32{scores()}
	segs := []segment{{tokens: []int32{4}, parent: -1, scored: 1}}
	if _, err := c.PrefillSegments(context.Background(), engine.TextPrompt([]int32{1, 2, 3}), segs, outs); err != nil {
		t.Fatal(err)
	}
	e.fed, e.posOf = nil, nil
	segs[0].tokens = []int32{7}
	if _, err := c.PrefillSegments(context.Background(), engine.TextPrompt([]int32{1, 2, 3}), segs, outs); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(e.fed, []int32{3, 7}) || !slices.Equal(e.posOf, []int{2, 3}) {
		t.Fatalf("fed %v at %v", e.fed, e.posOf)
	}
}

// What does not fit the prompt's last pass goes in a pass of its own.
func TestPrefillSegmentsFillThePasses(t *testing.T) {
	e := &branchEngine{}
	c := NewContext(branching(t, e), 0, 4096, time.Now, 0)
	ids := make([]int32, promptBatch-1)
	for i := range ids {
		ids[i] = int32(i + 1)
	}
	outs := [][]float32{scores(), scores()}
	segs := []segment{{tokens: []int32{100, 101, 102}, parent: -1, scored: 2}}
	if _, err := c.PrefillSegments(context.Background(), engine.TextPrompt(ids), segs, outs); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(e.widths, []int{promptBatch, 2}) {
		t.Fatalf("passes of %v", e.widths)
	}
	if outs[0][0] != 101 || outs[1][0] != 102 {
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
		[]weighing{{tail: []int32{3}, options: [][]int32{{5}, {6, 7}, {6, 8}}}}, nil, 10)
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
	}, nil, 8)
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

// A model that opens an empty thought before answering is weighed past it,
// on that question and on the ones after it. The thought follows the words
// already read, and only it is fed.
func TestWeighGoesPastTheEmptyThought(t *testing.T) {
	row := func(vals map[int]float32) []float32 {
		r := make([]float32, 12)
		for i := range r {
			r[i] = -30
		}
		for k, v := range vals {
			r[k] = v
		}
		return r
	}
	// After a question's words (3 or 4) the model opens a thought (9 10);
	// past it (10) it answers 5 rather than 6.
	e := &tableEngine{next: map[int32][]float32{
		3:  row(map[int]float32{9: 5, 5: 1, 6: 0}),
		4:  row(map[int]float32{9: 5, 5: 1, 6: 0}),
		10: row(map[int]float32{5: 2, 6: 0}),
	}}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	text := func(ids []int32) (engine.Prompt, error) { return engine.TextPrompt(ids), nil }
	got, _, err := c.Weigh(context.Background(), text, []int32{1, 2}, []weighing{
		{tail: []int32{3}, options: [][]int32{{5}, {6}}},
		{tail: []int32{4}, options: [][]int32{{5}, {6}}},
	}, []int32{9, 10}, 12)
	if err != nil {
		t.Fatal(err)
	}
	for q := range got {
		if want := logSoftmaxAt(e.next[10], 5); math.Abs(got[q][0]-want) > 1e-9 {
			t.Fatalf("question %d weighs 5 at %v, want %v past the thought", q, got[q][0], want)
		}
	}
	want := []int32{1, 2, 3, 9, 10, 4, 9, 10}
	if !slices.Equal(e.fed, want) {
		t.Fatalf("fed %v, want %v", e.fed, want)
	}
}

// branchTable is a tableEngine that carries branches. A table answers by the
// token fed alone, so a branch weighs what it would weigh one after another,
// and what these tests watch is how many passes it took.
type branchTable struct{ tableEngine }

func (e *branchTable) ForwardChunk(c engine.Chunk) [][]float32 { return e.chunk(c) }

// With branches, a question's options go in the pass of its words, and weigh
// what they weighed one after the other.
func TestWeighPutsAQuestionsOptionsInOnePass(t *testing.T) {
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
	e := &branchTable{tableEngine{next: map[int32][]float32{
		3: row(map[int]float32{5: 1, 6: 2}),
		6: row(map[int]float32{7: 0.5, 8: -0.5}),
	}}}
	c := NewContext(branching(t, e), 0, 4096, time.Now, 0)
	text := func(ids []int32) (engine.Prompt, error) { return engine.TextPrompt(ids), nil }
	got, fed, err := c.Weigh(context.Background(), text, []int32{1, 2},
		[]weighing{{tail: []int32{3}, options: [][]int32{{5}, {6, 7}, {6, 8}}}}, nil, 10)
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
	if !slices.Equal(e.widths, []int{5}) || fed != 5 {
		t.Fatalf("passes of %v, %d positions fed", e.widths, fed)
	}
}

// The empty thought is found and gone past the same way with branches: the
// question is read again with it, and the ones after it start with it.
func TestWeighGoesPastTheEmptyThoughtWithBranches(t *testing.T) {
	row := func(vals map[int]float32) []float32 {
		r := make([]float32, 12)
		for i := range r {
			r[i] = -30
		}
		for k, v := range vals {
			r[k] = v
		}
		return r
	}
	e := &branchTable{tableEngine{next: map[int32][]float32{
		3:  row(map[int]float32{9: 5, 5: 1, 6: 0}),
		4:  row(map[int]float32{9: 5, 5: 1, 6: 0}),
		10: row(map[int]float32{5: 2, 6: 0}),
	}}}
	c := NewContext(branching(t, e), 0, 4096, time.Now, 0)
	text := func(ids []int32) (engine.Prompt, error) { return engine.TextPrompt(ids), nil }
	got, _, err := c.Weigh(context.Background(), text, []int32{1, 2}, []weighing{
		{tail: []int32{3}, options: [][]int32{{5}, {6}}},
		{tail: []int32{4}, options: [][]int32{{5}, {6}}},
	}, []int32{9, 10}, 12)
	if err != nil {
		t.Fatal(err)
	}
	for q := range got {
		if want := logSoftmaxAt(e.next[10], 5); math.Abs(got[q][0]-want) > 1e-9 {
			t.Fatalf("question %d weighs 5 at %v, want %v past the thought", q, got[q][0], want)
		}
	}
	// Both questions at once; the thought turns up after the first one's
	// words, and both are read again with it, after the state's last position.
	if want := []int32{1, 2, 3, 4, 2, 3, 9, 10, 4, 9, 10}; !slices.Equal(e.fed, want) {
		t.Fatalf("fed %v, want %v", e.fed, want)
	}
}

// Every question about a state goes in the passes its positions fill, here
// one.
func TestWeighPutsEveryQuestionInOnePass(t *testing.T) {
	e := &branchTable{}
	c := NewContext(branching(t, e), 0, 4096, time.Now, 0)
	text := func(ids []int32) (engine.Prompt, error) { return engine.TextPrompt(ids), nil }
	_, fed, err := c.Weigh(context.Background(), text, []int32{1, 2, 3}, []weighing{
		{tail: []int32{7}, options: [][]int32{{1}, {2}}},
		{tail: []int32{8, 9}, options: [][]int32{{1}, {2, 3}}},
	}, nil, 8)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{1, 2, 3, 7, 8, 9, 2}; !slices.Equal(e.fed, want) || !slices.Equal(e.widths, []int{7}) || fed != 7 {
		t.Fatalf("fed %v in passes of %v, counted %d", e.fed, e.widths, fed)
	}
}
