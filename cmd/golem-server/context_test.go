package main

import (
	"testing"
	"time"
)

// An engine that records what it was fed and at which position, and whose
// hidden state is the token itself, so a test can see which state came back.
type recordingEngine struct {
	slot int

	fed    []int32
	posOf  []int
	resets int
}

// ForwardSlots is what the runner calls; the tests below drive one slot, and
// TestEachContextDrivesItsOwnSlot is what watches several.
func (e *recordingEngine) ForwardSlots(tokens []int32, slots, positions []int) [][]float32 {
	e.slot = slots[0]
	return e.ForwardBatch(tokens, positions[0])
}

func (e *recordingEngine) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	hidden := make([][]float32, len(tokens))
	for i, token := range tokens {
		e.fed = append(e.fed, token)
		e.posOf = append(e.posOf, startPos+i)
		hidden[i] = []float32{float32(token)}
	}
	return hidden
}

// The score of a state is the state itself, so a test can say which token the
// pass ended on.
func (e *recordingEngine) LogitsBatch(hidden [][]float32, out [][]float32) {
	for i, o := range out {
		clear(o)
		o[0] = hidden[i][0]
	}
}

func (e *recordingEngine) Reset() { e.resets++ }

// slot is the last one the context asked for, so a test can check that every
// pass names its own.
func (e *recordingEngine) UseSlot(i int) { e.slot = i }

func TestPrefillFeedsTheWholePromptTheFirstTime(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	fed, err := c.Prefill([]int32{1, 2, 3}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 3 || c.Pos() != 3 {
		t.Fatalf("fed %d, pos %d", fed, c.Pos())
	}
}

// The second request repeats the conversation and adds to it: only the addition
// is fed.
func TestPrefillReusesTheCommonPrefix(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	if _, err := c.Prefill([]int32{1, 2, 3}, scores()); err != nil {
		t.Fatal(err)
	}
	logits := scores()
	fed, err := c.Prefill([]int32{1, 2, 3, 4, 5}, logits)
	if err != nil {
		t.Fatal(err)
	}
	if fed != 2 {
		t.Fatalf("fed %d positions, want 2: the prefix was not reused", fed)
	}
	if logits[0] != 5 {
		t.Fatalf("the state scored came from token %v, want the last one", logits[0])
	}
	if c.Pos() != 5 {
		t.Fatalf("pos %d", c.Pos())
	}
	if e.posOf[3] != 3 || e.posOf[4] != 4 {
		t.Fatalf("positions %v: the continuation did not carry on", e.posOf)
	}
}

// A prompt identical to what is cached still has to feed its last position: the
// hidden state of a position already in the cache was not kept.
func TestPrefillOfAnIdenticalPromptFeedsOnePosition(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	if _, err := c.Prefill([]int32{1, 2, 3}, scores()); err != nil {
		t.Fatal(err)
	}
	fed, err := c.Prefill([]int32{1, 2, 3}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 1 {
		t.Fatalf("fed %d, want the last position only", fed)
	}
	if c.Pos() != 3 {
		t.Fatalf("pos %d", c.Pos())
	}
}

// A different conversation diverges early and is fed from the divergence.
func TestPrefillFeedsFromTheDivergence(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, time.Now, 0)
	if _, err := c.Prefill([]int32{1, 2, 3, 4}, scores()); err != nil {
		t.Fatal(err)
	}
	fed, err := c.Prefill([]int32{1, 2, 9}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 1 {
		t.Fatalf("fed %d, want the one position that differs", fed)
	}
	if c.Pos() != 3 {
		t.Fatalf("pos %d", c.Pos())
	}
}

// Rewinding a ring: the positions inside the window that the longer run
// overwrote have to be fed again, so prefill restarts a window early.
func TestRewindingAWindowRestartsAWindowEarly(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 4, 4096, time.Now, 0)
	if _, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6, 7, 8}, scores()); err != nil {
		t.Fatal(err)
	}
	fed, err := c.Prefill([]int32{1, 2, 3, 4, 5, 9}, scores())
	if err != nil {
		t.Fatal(err)
	}
	// The prefix is 5 long, the window is 4: positions 2..5 have to be fed
	// again, which is 4 positions counting the one that differs.
	if fed != 4 {
		t.Fatalf("fed %d, want the window's worth", fed)
	}
	if e.posOf[len(e.posOf)-1] != 5 || c.Pos() != 6 {
		t.Fatalf("positions %v, pos %d", e.posOf, c.Pos())
	}
}

// Appending never rewinds, even with a window.
func TestAppendingWithAWindowFeedsOnlyTheAddition(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 4, 4096, time.Now, 0)
	if _, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6}, scores()); err != nil {
		t.Fatal(err)
	}
	fed, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6, 7}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 1 {
		t.Fatalf("fed %d, want 1", fed)
	}
}

// The time to live drops what is held, so the next request starts over.
func TestTheTimeToLiveForgetsTheCache(t *testing.T) {
	clock := time.Unix(0, 0)
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, func() time.Time { return clock }, time.Minute)
	if _, err := c.Prefill([]int32{1, 2, 3}, scores()); err != nil {
		t.Fatal(err)
	}
	clock = clock.Add(2 * time.Minute)
	fed, err := c.Prefill([]int32{1, 2, 3, 4}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 4 {
		t.Fatalf("fed %d, want the whole conversation after the cache expired", fed)
	}
	if e.resets != 1 {
		t.Fatalf("%d resets", e.resets)
	}
}

// Inside the delay, nothing is forgotten.
func TestTheTimeToLiveKeepsTheCacheUntilItExpires(t *testing.T) {
	clock := time.Unix(0, 0)
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4096, func() time.Time { return clock }, time.Minute)
	if _, err := c.Prefill([]int32{1, 2, 3}, scores()); err != nil {
		t.Fatal(err)
	}
	clock = clock.Add(30 * time.Second)
	if fed, _ := c.Prefill([]int32{1, 2, 3, 4}, scores()); fed != 1 {
		t.Fatalf("fed %d, want 1", fed)
	}
	if e.resets != 0 {
		t.Fatalf("%d resets", e.resets)
	}
}

func TestAPromptPastTheContextIsRefused(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 0, 4, time.Now, 0)
	if _, err := c.Prefill([]int32{1, 2, 3, 4, 5}, scores()); err == nil {
		t.Fatal("a prompt past the context should be refused rather than wrap the cache")
	}
}

// running is a runner over one engine, owning it for the test's lifetime.
// Every conversation in these tests goes through one, because in the server
// every conversation does.
func running(tb testing.TB, e Engine) *Runner {
	r := NewRunner(e)
	stop := make(chan struct{})
	go r.Run(stop)
	tb.Cleanup(func() { close(stop) })
	return r
}

// scores is a logits buffer for a test that does not read it. The fake engines
// score into it and nothing looks: what these tests watch is what was fed.
func scores() []float32 { return make([]float32, 8) }
