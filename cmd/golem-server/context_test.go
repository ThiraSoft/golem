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
	widths []int // how many positions each pass carried
	resets int
}

// ForwardSlots is what the runner calls; the tests below drive one slot, and
// TestEachContextDrivesItsOwnSlot is what watches several.
func (e *recordingEngine) ForwardSlots(tokens []int32, slots, positions []int) [][]float32 {
	e.slot = slots[0]
	e.widths = append(e.widths, len(tokens))
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

func (e *recordingEngine) Logits(hidden []float32, out []float32) {
	e.LogitsBatch([][]float32{hidden}, [][]float32{out})
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

// Rewinding a ring of exactly the window: the longer run wrote positions 4..7
// over the slots of 0..3. Resuming at 5 would read 2..4 from those slots, and
// resuming a window early at 2 would still read 0 and 1 from them, so the
// whole prompt is fed again.
func TestRewindingAWindowFeedsEveryOverwrittenPosition(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 4, 4096, time.Now, 0)
	if _, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6, 7, 8}, scores()); err != nil {
		t.Fatal(err)
	}
	fed, err := c.Prefill([]int32{1, 2, 3, 4, 5, 9}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 6 {
		t.Fatalf("fed %d, want the whole prompt", fed)
	}
	if e.posOf[len(e.posOf)-1] != 5 || c.Pos() != 6 {
		t.Fatalf("positions %v, pos %d", e.posOf, c.Pos())
	}
}

// A ring holds the window and a pass more, as gemma's does. Another
// conversation on the same prefix went to position 13 and overwrote the slots
// of 0..5. Restarting a window early, at 3, read positions 0..2 from slots that
// held 8..10: the village's characters, sharing a long lore, answered with the
// keys of whoever had the slot before them.
func TestRewindingALargerRingFeedsWhatTheOtherConversationOverwrote(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 4, 4096, time.Now, 0)
	c.SetRing(8)
	first := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	if _, err := c.Prefill(first, scores()); err != nil {
		t.Fatal(err)
	}
	fed, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6, 99}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 7 {
		t.Fatalf("fed %d, want the whole prompt: its window reads slots the other run overwrote", fed)
	}
	if e.posOf[len(first)] != 0 {
		t.Fatalf("the second prompt was fed from position %d", e.posOf[len(first)])
	}
}

// The same ring, when the other conversation stopped short of wrapping: every
// slot still holds its own position, and only what differs is fed.
func TestRewindingARingThatDidNotWrapFeedsOnlyTheDivergence(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 4, 4096, time.Now, 0)
	c.SetRing(8)
	if _, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6, 7}, scores()); err != nil {
		t.Fatal(err)
	}
	fed, err := c.Prefill([]int32{1, 2, 3, 4, 5, 99}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 1 {
		t.Fatalf("fed %d, want the one position that differs", fed)
	}
}

// Tokens drawn one at a time are in the ring too: a conversation that grew
// past the ring by generation is rewound as if it had been prompted that far.
func TestDrawnTokensCountAsWrittenToTheRing(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(running(t, e), 4, 4096, time.Now, 0)
	c.SetRing(8)
	if _, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6}, scores()); err != nil {
		t.Fatal(err)
	}
	for id := int32(7); id <= 14; id++ {
		c.Advance(id, scores())
	}
	fed, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6, 99}, scores())
	if err != nil {
		t.Fatal(err)
	}
	if fed != 7 {
		t.Fatalf("fed %d, want the whole prompt", fed)
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

// A prompt is cut into passes of whatever width the model carries well, and
// where the model is decides that: thirty-two on the processor, where past
// sixty-four the activations stop fitting in the caches, and two hundred and
// fifty-six on a card, which is idle at thirty-two. Nothing else about a chunk
// changes with it.
func TestAPromptIsCutToThePassWidth(t *testing.T) {
	ids := make([]int32, 600)
	for i := range ids {
		ids[i] = int32(i + 1)
	}
	for _, one := range []struct {
		name   string
		device bool
		want   int
	}{
		{"processor", false, promptBatch},
		{"card", true, devicePassWidth},
	} {
		t.Run(one.name, func(t *testing.T) {
			e := &recordingEngine{}
			r := running(t, e)
			if one.device {
				r.OnDevice()
			}
			if got := r.PassWidth(); got != one.want {
				t.Fatalf("the runner carries %d positions a pass, want %d", got, one.want)
			}
			c := NewContext(r, 0, 4096, time.Now, 0)
			if _, err := c.Prefill(ids, scores()); err != nil {
				t.Fatal(err)
			}
			passes := (len(ids) + one.want - 1) / one.want
			if len(e.widths) != passes {
				t.Errorf("%d passes for %d positions, want %d", len(e.widths), len(ids), passes)
			}
			if len(e.widths) > 0 && e.widths[0] != one.want {
				t.Errorf("the first pass carried %d positions, want %d", e.widths[0], one.want)
			}
			if n := len(e.fed); n != len(ids) {
				t.Errorf("%d positions fed, want %d", n, len(ids))
			}
		})
	}
}

// Every picture has the same placeholder tokens: a new one is not the one the
// cache read, and the prefix stops where it starts.
func TestCommonStopsAtAnotherPicture(t *testing.T) {
	ids := []int32{1, 7, 7, 2}
	red, blue := []float32{1, 0}, []float32{0, 1}
	held := [][]float32{nil, red, red, nil}
	for _, c := range []struct {
		name   string
		embeds [][]float32
		want   int
	}{
		{"the same rows", [][]float32{nil, red, red, nil}, 4},
		{"equal rows, other slices", [][]float32{nil, {1, 0}, {1, 0}, nil}, 4},
		{"another picture", [][]float32{nil, blue, blue, nil}, 1},
		{"its second half changed", [][]float32{nil, red, blue, nil}, 1},
		{"text where a picture was", nil, 1},
	} {
		if got := common(ids, held, ids, c.embeds); got != c.want {
			t.Errorf("%s: %d positions shared, want %d", c.name, got, c.want)
		}
	}
	if got := common(ids, nil, ids, nil); got != 4 {
		t.Errorf("text alone: %d shared, want 4", got)
	}
}

// A recurrent model's cache is a state that has read every position held, so
// it can be continued but never rewound. The three cases a prompt can be in.
func TestARecurrentCacheIsContinuedOrStartedAgain(t *testing.T) {
	for _, tc := range []struct {
		name   string
		second []int32
		fed    int
		resets int
	}{
		// The conversation grew: only the addition is read.
		{"continued", []int32{1, 2, 3, 4, 5}, 2, 0},
		// The same prompt again: feeding its last position a second time
		// would read it twice into the state, so the slot starts over.
		{"repeated", []int32{1, 2, 3}, 3, 1},
		// It parts from what is held: there is no going back to 1, 2.
		{"rewound", []int32{1, 2, 9, 9}, 4, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := &recordingEngine{}
			c := NewContext(running(t, e), 0, 4096, time.Now, 0)
			c.SetRecurrent(true)
			if _, err := c.Prefill([]int32{1, 2, 3}, scores()); err != nil {
				t.Fatal(err)
			}
			fed, err := c.Prefill(tc.second, scores())
			if err != nil {
				t.Fatal(err)
			}
			if fed != tc.fed || e.resets != tc.resets {
				t.Fatalf("fed %d with %d resets, want %d and %d", fed, e.resets, tc.fed, tc.resets)
			}
			if got := e.posOf[len(e.posOf)-fed]; got != len(tc.second)-fed {
				t.Fatalf("the feed began at position %d, want %d", got, len(tc.second)-fed)
			}
		})
	}
}
