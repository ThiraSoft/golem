package nn

// The workers, and why they spin.
//
// A token is three hundred parallel sections long, each a few hundred
// microseconds apart, and each ends with every core waiting on the slowest.
// Handing those sections to fresh goroutines makes the runtime park and wake
// its threads three hundred times a token, and a wake-up costs tens of
// microseconds — measured on this engine, more than a tenth of the time a token
// takes.
//
// So the workers are made once and kept. Between sections they spin on a
// counter for a few hundred microseconds, which is longer than the gap between
// two sections during generation and short enough that an idle process settles
// down on its own: after that they block on a channel, and the next section
// pays the wake-up it would have paid anyway.
//
// Spinning is only ever right when there is a core to spin on. A worker that
// spins while another goroutine wants the same processor does not wait faster,
// it stops the work from happening: with GOMAXPROCS at one and eight workers
// spinning, a frame of the audio decoder took ten times longer than it does with
// no parallelism at all. So the pool asks the runtime how many processors it may
// use, splits the work that many ways, and lets the workers it did not use park
// at once rather than spin.

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// spinRounds is how long a worker waits before blocking, counted in idle
// iterations: a few milliseconds, which is longer than the gap between two
// sections of a token and short enough that a process left alone stops burning
// a core almost at once.
const spinRounds = 200000

type workers struct {
	count int

	// The section being run. Published by the sequence number: a worker that
	// sees a new sequence sees the fields written before it.
	work      func(start, end int)
	n         int
	grain     int
	cursor    atomic.Int64
	sequence  atomic.Uint64
	remaining atomic.Int64

	wake     []chan struct{}
	sleeping []atomic.Bool
	// spin says whether a worker that finds nothing to do should wait on the
	// counter or block. It is false whenever the last section used fewer
	// workers than the pool holds, which is exactly when spinning would take a
	// processor from someone who needs it.
	spin atomic.Bool
}

var pool = newWorkers()

func newWorkers() *workers {
	count := runtime.NumCPU()
	w := &workers{
		count:    count,
		wake:     make([]chan struct{}, count),
		sleeping: make([]atomic.Bool, count),
	}
	for i := 1; i < count; i++ {
		w.wake[i] = make(chan struct{}, 1)
		go w.serve(i)
	}
	return w
}

// serve is one worker's whole life: wait for a sequence it has not run, run its
// share of it, and wait again.
func (w *workers) serve(index int) {
	last := uint64(0)
	for {
		spins := 0
		for w.sequence.Load() == last {
			spins++
			if spins > spinRounds || !w.spin.Load() {
				w.sleeping[index].Store(true)
				// The section may have been published between the test and the
				// flag; look again before actually blocking.
				if w.sequence.Load() != last {
					w.sleeping[index].Store(false)
					break
				}
				<-w.wake[index]
				w.sleeping[index].Store(false)
				spins = 0
			}
			// A bare spin, not a yield: the workers are as many as the cores,
			// and asynchronous preemption is what keeps the runtime able to
			// stop them.
			spinPause()
		}
		last = w.sequence.Load()

		w.consume()
		w.remaining.Add(-1)
	}
}

// consume takes ranges of tasks until there are none left.
//
// The ranges are handed out rather than divided in advance: rows of a matrix
// are not all equally expensive — some are already in cache, some are not — and
// a core that finished early is worth more taking the next range than waiting
// at the barrier for one that did not.
func (w *workers) consume() {
	for {
		start := int(w.cursor.Add(int64(w.grain))) - w.grain
		if start >= w.n {
			return
		}
		w.work(start, min(start+w.grain, w.n))
	}
}

// run spreads n tasks over `chunks` chunks and returns when all are done. The
// caller runs the first chunk itself, which is both one fewer hand-off and the
// whole of the work when chunks is one.
func (w *workers) run(n, chunks int, work func(start, end int)) {
	if chunks <= 1 {
		work(0, n)
		return
	}
	// Four ranges per core: enough for a late core to catch up, few enough that
	// the counter they share is not what they queue on.
	grain := (n + chunks*4 - 1) / (chunks * 4)
	if grain < 1 {
		grain = 1
	}
	w.work, w.n, w.grain = work, n, grain
	w.cursor.Store(0)
	w.spin.Store(chunks >= w.count)
	w.remaining.Store(int64(w.count - 1))
	w.sequence.Add(1)
	for i := 1; i < w.count; i++ {
		if w.sleeping[i].Load() {
			select {
			case w.wake[i] <- struct{}{}:
			default:
			}
		}
	}

	w.consume()

	for w.remaining.Load() != 0 {
		if w.spin.Load() {
			spinPause()
			continue
		}
		runtime.Gosched()
	}
	w.work = nil
}

// InParallel spreads `n` independent tasks over the available cores, provided
// the total work justifies it. It is offered to the packages that have
// splittable work to hand off.
//
// The sections are not reentrant: the work function must not itself call
// InParallel, which nothing in this engine does — the split always happens at
// the outermost loop of a product.
var parallelMutex sync.Mutex

func InParallel(n, totalWork int, work func(start, end int)) {
	// However many workers the pool holds, the runtime decides how many
	// processors this program may use, and that number can change while it
	// runs. Splitting the work more ways than there are processors makes every
	// piece wait for the others through the same core.
	chunks := min(pool.count, runtime.GOMAXPROCS(0))
	if chunks > n {
		chunks = n
	}
	if totalWork < parallelThreshold {
		chunks = 1
	}
	if chunks <= 1 {
		work(0, n)
		return
	}
	parallelMutex.Lock()
	pool.run(n, chunks, work)
	parallelMutex.Unlock()
}
