package nn

import (
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
)

// Every task is handed out exactly once, whatever the split.
func TestInParallelCoversEveryTask(t *testing.T) {
	for _, n := range []int{1, 7, 8, 9, 1000} {
		done := make([]int32, n)
		InParallel(n, 1<<40, func(start, end int) {
			for i := start; i < end; i++ {
				atomic.AddInt32(&done[i], 1)
			}
		})
		for i, count := range done {
			if count != 1 {
				t.Fatalf("n=%d: task %d ran %d times", n, i, count)
			}
		}
	}
}

// A program the runtime has given one processor gets one piece of work, run on
// the caller. Splitting it further would make the pieces wait for each other
// through the same core — and the workers spin while they wait, so the cost is
// not a little parallel overhead but an order of magnitude.
func TestInParallelObeysGOMAXPROCS(t *testing.T) {
	previous := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(previous)

	var pieces atomic.Int64
	InParallel(1000, 1<<40, func(start, end int) { pieces.Add(1) })
	if got := pieces.Load(); got != 1 {
		t.Fatalf("with one processor the work was cut into %d pieces, want 1", got)
	}
}

// The ranges shrink as the work runs out.
//
// This is what keeps a slow core from setting the barrier: whatever range it
// takes last, it is one of the small ones, so everyone else waits a few rows
// rather than a whole even share. Cores of different speeds are the normal case
// on arm64 — performance and efficiency cores in the same chip — and there the
// difference is two or three times, not a few percent.
func TestInParallelHandsOutShrinkingRanges(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("one processor: the work is not split at all")
	}
	const n = 4096

	var mutex sync.Mutex
	sizes := map[int]int{}
	InParallel(n, 1<<40, func(start, end int) {
		mutex.Lock()
		sizes[start] = end - start
		mutex.Unlock()
	})

	var order []int
	for start := range sizes {
		order = append(order, start)
	}
	sort.Ints(order)

	first, last := sizes[order[0]], sizes[order[len(order)-1]]
	for i := 1; i < len(order); i++ {
		if sizes[order[i]] > sizes[order[i-1]] {
			t.Fatalf("range at %d is %d, larger than the %d before it",
				order[i], sizes[order[i]], sizes[order[i-1]])
		}
	}
	if last*4 > first {
		t.Fatalf("the last range is %d against a first of %d: the tail does not shrink", last, first)
	}
}
