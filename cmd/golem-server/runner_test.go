package main

import (
	"strings"
	"sync"
	"testing"
	"time"
)

// One model, driven from one place. Several slots mean several conversations
// in flight, and the weights, the scratch and the caches behind them are one
// set of buffers: two requests that touched the model at once would be reading
// what the other is writing. Under -race, this is what says so.
func TestTwoRequestsDoNotTouchTheModelAtOnce(t *testing.T) {
	s := newTestServerWithSlots(t, []string{"one", "two", "three"}, 2)

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			body := `{"messages":[{"role":"user","content":"` + strings.Repeat("word ", 4+i) + `"}]}`
			if w := post(t, s, body); w.Code != 200 {
				t.Errorf("%d: %s", w.Code, w.Body)
			}
		}(i)
	}
	wg.Wait()
}

// A batching engine that says how many tokens each pass carried, and how many
// conversations they came from.
type countingEngine struct {
	*scriptedEngine
	mu     sync.Mutex
	widest int
	slots  int
}

func (e *countingEngine) ForwardSlots(tokens []int32, slots, positions []int) [][]float32 {
	e.mu.Lock()
	seen := map[int]bool{}
	for _, s := range slots {
		seen[s] = true
	}
	if len(tokens) > e.widest {
		e.widest = len(tokens)
	}
	if len(seen) > e.slots {
		e.slots = len(seen)
	}
	e.mu.Unlock()
	// Long enough that the other conversation reaches the runner while this
	// one is inside the model, which is what a real pass does by itself.
	time.Sleep(2 * time.Millisecond)
	return e.scriptedEngine.ForwardSlots(tokens, slots, positions)
}

// Two conversations drawing at once go through the model together. That is the
// whole of continuous batching: one read of the weights, two tokens out.
func TestTwoConversationsShareOnePass(t *testing.T) {
	script := make([]string, 0, 40)
	for i := 0; i < 40; i++ {
		script = append(script, "word")
	}
	g, v := newGenerator(t, script, 24)
	e := &countingEngine{scriptedEngine: g.ctx.runner.engine.(*scriptedEngine)}
	runner := running(t, e)
	slots := make([]*slot, 2)
	for i := range slots {
		ctx := NewSlotContext(runner, i, 0, 4096, time.Now, 0)
		gen := NewGenerator(ctx, v, wordTemplate{}, len(g.logits), 24)
		slots[i] = &slot{index: i, ctx: ctx, gen: gen}
	}
	s := NewServer(NewPool(slots, time.Now), v, "test-model", wordTemplate{}, greedy())

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			body := `{"messages":[{"role":"user","content":"` + strings.Repeat("word ", 3+i) + `"}]}`
			if w := post(t, s, body); w.Code != 200 {
				t.Errorf("%d: %s", w.Code, w.Body)
			}
		}(i)
	}
	wg.Wait()

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.slots < 2 {
		t.Errorf("no pass carried more than %d conversation, and two were drawing at once", e.slots)
	}
}
