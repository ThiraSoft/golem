package main

// Several conversations at once, over one set of weights.
//
// A slot is a KV cache and the tokens it holds. The model is still driven by
// one request at a time — there is one scratch and one set of weights, and
// eight cores are already busy with the request that holds them — so what
// slots change is not the throughput but the cache: two clients talking to the
// same server stop overwriting each other's prefix, and each of them keeps
// paying one exchange per turn rather than the whole conversation.
//
// Which slot answers is decided the way llama.cpp's server decides it: the
// free slot whose held tokens share the longest prefix with this prompt, as a
// fraction of the prompt, provided it clears a threshold; failing that, the
// one used longest ago. When none is free the request waits, and a client that
// hangs up while waiting takes its request with it.

import (
	"context"
	"sync"
	"time"
)

// similarity is the fraction of a prompt a slot must already hold to be chosen
// for it. llama.cpp's default is a tenth, and the reasoning carries: below
// that the prefix saves little, and taking the slot costs whoever held it
// everything they had.
const similarity = 0.1

type slot struct {
	index int
	ctx   *Context
	gen   *Generator

	busy bool
	last time.Time
}

// Pool hands slots out and takes them back.
type Pool struct {
	mu    sync.Mutex
	slots []*slot
	free  chan struct{} // one token per free slot
	now   func() time.Time
}

func NewPool(slots []*slot, now func() time.Time) *Pool {
	p := &Pool{slots: slots, free: make(chan struct{}, len(slots)), now: now}
	for range slots {
		p.free <- struct{}{}
	}
	return p
}

// Len is how many conversations the pool holds at once.
func (p *Pool) Len() int { return len(p.slots) }

// Acquire takes the slot that fits this prompt best, waiting for one if they
// are all busy. It reports how long the wait was, so that a queue is visible
// in the log rather than hidden inside a slow answer.
func (p *Pool) Acquire(ctx context.Context, ids []int32) (*slot, time.Duration, error) {
	start := p.now()
	select {
	case <-p.free:
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	}
	waited := p.now().Sub(start)

	p.mu.Lock()
	defer p.mu.Unlock()
	s := p.pick(ids)
	s.busy = true
	s.last = p.now()
	return s, waited, nil
}

// pick chooses among the free slots. A token was taken from p.free, so at
// least one of them is.
func (p *Pool) pick(ids []int32) *slot {
	var best *slot
	bestShare := similarity
	for _, s := range p.slots {
		if s.busy || len(s.ctx.held) == 0 {
			continue
		}
		share := float64(commonPrefix(s.ctx.held, ids)) / float64(len(ids))
		if share > bestShare {
			best, bestShare = s, share
		}
	}
	if best != nil {
		return best
	}
	// Nothing shared worth keeping. A slot holding no conversation costs
	// nothing to take, so it goes before one that would be thrown away —
	// llama.cpp goes straight to the least recently used, which reaches the
	// same slot whenever an empty one has been idle longest, and does not when
	// it has not.
	for _, s := range p.slots {
		if !s.busy && len(s.ctx.held) == 0 {
			return s
		}
	}
	// Failing that, the conversation nobody has come back to in longest.
	for _, s := range p.slots {
		if s.busy {
			continue
		}
		if best == nil || s.last.Before(best.last) {
			best = s
		}
	}
	return best
}

// Release gives a slot back.
func (p *Pool) Release(s *slot) {
	p.mu.Lock()
	s.busy = false
	s.last = p.now()
	p.mu.Unlock()
	p.free <- struct{}{}
}

func commonPrefix(a, b []int32) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}
