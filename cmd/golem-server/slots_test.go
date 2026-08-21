package main

import (
	"context"
	"testing"
	"time"
)

// A pool of n slots over one recording engine, each holding what it is given.
func poolOfHeld(t *testing.T, held ...[]int32) (*Pool, *recordingEngine) {
	t.Helper()
	e := &recordingEngine{}
	slots := make([]*slot, len(held))
	for i := range held {
		c := NewSlotContext(e, i, 0, 64, time.Now, 0)
		c.held = held[i]
		slots[i] = &slot{index: i, ctx: c, last: time.Unix(int64(i), 0)}
	}
	return NewPool(slots, time.Now), e
}

// The slot that already holds most of this prompt is the one that answers it.
func TestAcquireTakesTheLongestSharedPrefix(t *testing.T) {
	pool, _ := poolOfHeld(t,
		[]int32{1, 2, 3, 4, 5, 6, 7, 8},
		[]int32{1, 2, 9},
	)
	got, _, err := pool.Acquire(context.Background(), []int32{1, 2, 3, 4, 5, 6, 7, 99})
	if err != nil {
		t.Fatal(err)
	}
	if got.index != 0 {
		t.Errorf("slot %d answered, want the one holding seven of the eight tokens", got.index)
	}
}

// A prompt sharing too little with any of them takes the slot used longest
// ago, rather than destroying a conversation that is still worth something.
func TestAcquireFallsBackToLeastRecentlyUsed(t *testing.T) {
	pool, _ := poolOfHeld(t,
		[]int32{1, 2, 3, 4, 5, 6, 7, 8},
		[]int32{1, 2, 3, 4, 5, 6, 7, 8},
	)
	pool.slots[1].last = time.Unix(0, 0)
	pool.slots[0].last = time.Unix(100, 0)

	got, _, err := pool.Acquire(context.Background(), []int32{50, 51, 52, 53})
	if err != nil {
		t.Fatal(err)
	}
	if got.index != 1 {
		t.Errorf("slot %d answered, want the one used longest ago", got.index)
	}
}

// An empty slot is taken before a conversation is thrown away.
func TestAcquirePrefersAnEmptySlot(t *testing.T) {
	pool, _ := poolOfHeld(t, []int32{1, 2, 3, 4}, nil)
	pool.slots[1].last = time.Unix(1000, 0) // and used most recently, at that
	got, _, err := pool.Acquire(context.Background(), []int32{9, 9, 9})
	if err != nil {
		t.Fatal(err)
	}
	if got.index != 1 {
		t.Errorf("slot %d answered, want the empty one", got.index)
	}
}

// With every slot busy the request waits, and the wait is reported so that a
// queue is visible rather than hidden inside a slow answer.
func TestAcquireWaitsForABusySlot(t *testing.T) {
	pool, _ := poolOfHeld(t, []int32{1})
	held, _, err := pool.Acquire(context.Background(), []int32{1})
	if err != nil {
		t.Fatal(err)
	}

	done := make(chan time.Duration)
	go func() {
		_, waited, err := pool.Acquire(context.Background(), []int32{1})
		if err != nil {
			t.Error(err)
		}
		done <- waited
	}()

	select {
	case <-done:
		t.Fatal("a second request took a slot that was busy")
	case <-time.After(20 * time.Millisecond):
	}
	pool.Release(held)
	if waited := <-done; waited < 10*time.Millisecond {
		t.Errorf("the wait was reported as %s, and it was at least twenty milliseconds", waited)
	}
}

// A client that hangs up while queued takes its request with it.
func TestAcquireGivesUpWhenTheClientDoes(t *testing.T) {
	pool, _ := poolOfHeld(t, []int32{1})
	if _, _, err := pool.Acquire(context.Background(), []int32{1}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()
	got, _, err := pool.Acquire(ctx, []int32{1})
	if err == nil {
		t.Fatalf("slot %d was handed to a client that had gone", got.index)
	}
}

// Every pass names its own cache, because between two of them another
// conversation may have run.
func TestEachContextDrivesItsOwnSlot(t *testing.T) {
	e := &recordingEngine{}
	first := NewSlotContext(e, 0, 0, 64, time.Now, 0)
	second := NewSlotContext(e, 1, 0, 64, time.Now, 0)

	if _, _, err := first.Prefill([]int32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if e.slot != 0 {
		t.Errorf("the engine was pointed at slot %d, want 0", e.slot)
	}
	if _, _, err := second.Prefill([]int32{3, 4}); err != nil {
		t.Fatal(err)
	}
	if e.slot != 1 {
		t.Errorf("the engine was pointed at slot %d, want 1", e.slot)
	}
	first.Advance(5)
	if e.slot != 0 {
		t.Errorf("advancing wrote into slot %d, want 0", e.slot)
	}
}
