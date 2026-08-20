package nn

import (
	"runtime"
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
