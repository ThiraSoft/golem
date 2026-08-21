package main

// The one place the model is touched.
//
// A conversation does not call the model. It asks the runner for a pass and
// waits for the hidden state that comes back, and the runner — alone in its
// goroutine — is what reads the weights. Two things follow from that, and the
// second is the point of it.
//
// The weights, the scratch and the caches are one set of buffers, so exactly
// one goroutine may be inside the model at a time. That used to be a mutex
// around a whole answer, which made a second request wait for the first.
//
// And what is waiting at the moment a pass is built goes into it together.
// Reading a gigabyte of weights to advance one conversation by one token, when
// four conversations each want one token, is four times the memory traffic for
// the same arithmetic; ForwardSlots carries all four in one read. That is
// llama.cpp's continuous batching, and it is why generation for several
// clients at once costs barely more than for one.
//
// The gather window is what makes it happen. When a pass is ready but other
// conversations are still between two of theirs — sampling, or being handed a
// token — the runner waits a fraction of what a pass costs, and no longer:
// enough for the others to arrive, little enough that a lone client cannot
// feel it.

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// debugBatches says what went into each pass, for measuring how well the
// conversations in flight are actually meeting.
var debugBatches = os.Getenv("GOLEM_DEBUG_BATCHES") != ""

func distinct(slots []int) map[int]bool {
	seen := map[int]bool{}
	for _, s := range slots {
		seen[s] = true
	}
	return seen
}

// Engine is what the runner drives. Nothing else in this command holds one.
type Engine interface {
	ForwardSlots(tokens []int32, slots, positions []int) [][]float32
	LogitsBatch(hidden [][]float32, out [][]float32)
	UseSlot(i int)
	Reset()
}

// pass is one conversation's request for a forward pass, and for the scores of
// what it ends on. The two travel together because they always did: every pass
// but the intermediate chunks of a long prompt is followed by scoring its last
// state, and the head is a matrix large enough that reading it once for four
// conversations rather than four times is most of what batching them is worth.
type pass struct {
	slot      int
	tokens    []int32
	positions []int
	logits    []float32     // where to score the last token, or nil for a chunk
	reply     chan struct{} // closed once the pass has run
}

// aside is anything else the model has to do, which cannot overlap a pass:
// scoring a hidden state, or forgetting a conversation.
type aside struct {
	run  func()
	done chan struct{}
}

type Runner struct {
	engine Engine
	passes chan *pass
	asides chan *aside

	// waiting is how many conversations are in flight, which says whether a
	// pass in hand is all there is or the first of several.
	mu      sync.Mutex
	waiting int
	// span is what the last pass took, and the gather window is a fraction of
	// it: a model drawing a token in 45ms can afford to wait; one drawing it
	// in 700µs cannot.
	span time.Duration
}

// budget is how many positions go through the model in one pass. It is the
// same thirty-two a prompt is read in — past sixty-four the activations stop
// fitting in the caches, whether they belong to one conversation or four.
const budget = promptBatch

// gatherShare is the fraction of a pass the runner will spend waiting for
// company before going without it.
const gatherShare = 8

func NewRunner(e Engine) *Runner {
	return &Runner{
		engine: e,
		passes: make(chan *pass),
		asides: make(chan *aside),
	}
}

// Enter and Leave bracket a conversation, so that the runner knows how many
// passes to expect before it stops waiting for them.
func (r *Runner) Enter() { r.mu.Lock(); r.waiting++; r.mu.Unlock() }
func (r *Runner) Leave() { r.mu.Lock(); r.waiting--; r.mu.Unlock() }

func (r *Runner) inFlight() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.waiting
}

func (r *Runner) lastSpan() time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.span
}

func (r *Runner) window() time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	w := r.span / gatherShare
	if w > 2*time.Millisecond {
		w = 2 * time.Millisecond
	}
	return w
}

// Forward advances one conversation's cache by a run of tokens, and scores the
// last of them into logits. A nil logits is a chunk in the middle of a prompt,
// which nobody reads the scores of.
func (r *Runner) Forward(slot int, tokens []int32, positions []int, logits []float32) {
	p := &pass{slot: slot, tokens: tokens, positions: positions, logits: logits,
		reply: make(chan struct{})}
	r.passes <- p
	<-p.reply
}

// Reset forgets what one slot holds.
func (r *Runner) Reset(slot int) {
	r.do(func() {
		r.engine.UseSlot(slot)
		r.engine.Reset()
	})
}

func (r *Runner) do(f func()) {
	a := &aside{run: f, done: make(chan struct{})}
	r.asides <- a
	<-a.done
}

// Run owns the model until the channel closes. Everything above is a request
// to this loop.
func (r *Runner) Run(stop <-chan struct{}) {
	var held []*pass // taken from the channel and not yet run
	for {
		batch := held
		held = nil
		if len(batch) == 0 {
			select {
			case p := <-r.passes:
				batch = append(batch, p)
			case a := <-r.asides:
				a.run()
				close(a.done)
				continue
			case <-stop:
				return
			}
		}
		batch, held = r.gather(batch, held)
		r.run(batch)
	}
}

// gather waits, briefly, for the passes of the other conversations in flight,
// and returns what goes through the model now and what waits for the next one.
func (r *Runner) gather(batch, held []*pass) (now, later []*pass) {
	deadline := time.After(r.window())
	size := positions(batch)
	for size < budget && len(batch)+len(held) < r.inFlight() {
		select {
		case p := <-r.passes:
			if size+len(p.tokens) > budget {
				held = append(held, p)
				continue
			}
			batch = append(batch, p)
			size += len(p.tokens)
		case a := <-r.asides:
			// Somebody is scoring what the last pass gave them, which is the
			// step before they ask for another. Serve it and keep gathering.
			a.run()
			close(a.done)
		case <-deadline:
			return batch, held
		}
	}
	return batch, held
}

// run carries a batch through the model and hands each conversation the state
// of its own last token.
func (r *Runner) run(batch []*pass) {
	var tokens []int32
	var slots, at []int
	for _, p := range batch {
		tokens = append(tokens, p.tokens...)
		at = append(at, p.positions...)
		for range p.tokens {
			slots = append(slots, p.slot)
		}
	}
	if debugBatches {
		fmt.Fprintf(os.Stderr, "pass: %d tokens from %d conversation(s), last took %s, window %s\n",
			len(tokens), len(distinct(slots)), r.lastSpan().Round(time.Millisecond), r.window())
	}
	start := time.Now()
	states := r.engine.ForwardSlots(tokens, slots, at)

	// And one read of the head for everything in the batch that is waiting to
	// be scored.
	var last [][]float32
	var outs [][]float32
	end := 0
	for _, p := range batch {
		end += len(p.tokens)
		if p.logits != nil {
			last = append(last, states[end-1])
			outs = append(outs, p.logits)
		}
	}
	if len(last) > 0 {
		r.engine.LogitsBatch(last, outs)
	}
	took := time.Since(start)

	r.mu.Lock()
	r.span = took
	r.mu.Unlock()

	for _, p := range batch {
		close(p.reply)
	}
}

func positions(batch []*pass) int {
	n := 0
	for _, p := range batch {
		n += len(p.tokens)
	}
	return n
}
