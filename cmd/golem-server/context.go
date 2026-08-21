package main

// What the KV cache holds, across requests that carry no state.
//
// /v1/chat/completions is stateless: every request sends the whole
// conversation. The cache is not — it is the whole reason a turn costs one turn
// rather than the conversation. So the server remembers which tokens are in the
// cache, and a request only pays for what its prompt does not share with them.
//
// One rule earns its own paragraph. A sliding-window block stores its keys in a
// ring of exactly the window, so writing past position P and then rewinding to
// P overwrites the slots of positions P-W+1 … Q-W, which are still visible from
// P. Rewinding therefore restarts a window early rather than at P. Rewriting
// those positions is idempotent for the global blocks. Appending — which is
// what a conversation growing by one exchange does — rewinds nothing and costs
// nothing.

import (
	"fmt"
	"time"
)

// Engine is the part of a model the server drives. engine.Open supplies
// whatever satisfies it, whichever architecture the file declared.
type Engine interface {
	ForwardBatch(tokens []int32, startPos int) [][]float32
	Logits(hidden, out []float32)
	Reset()
	UseSlot(i int)
}

// Vocabulary is the part of bpe.Vocab the server uses.
type Vocabulary interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Piece(id int32, special bool) string
	IsEOG(id int32) bool
}

// promptBatch is how many positions go through the model together. Thirty-two
// is what cmd/golem-cli measured: the gain is flat by sixteen, and past sixty-four
// the activations stop fitting in the caches.
const promptBatch = 32

type Context struct {
	engine     Engine
	slot       int // which of the engine's caches this context is
	window     int // the largest sliding window; 0 when every block is global
	maxContext int
	now        func() time.Time
	ttl        time.Duration

	held []int32 // the tokens the cache holds, position by position
	last time.Time
}

func NewContext(e Engine, window, maxContext int, now func() time.Time, ttl time.Duration) *Context {
	return &Context{engine: e, window: window, maxContext: maxContext, now: now, ttl: ttl}
}

// NewSlotContext is the same, for one of several caches the engine holds.
func NewSlotContext(e Engine, slot, window, maxContext int, now func() time.Time, ttl time.Duration) *Context {
	c := NewContext(e, window, maxContext, now, ttl)
	c.slot = slot
	return c
}

// use points the engine at this context's cache. Every pass goes through it,
// because between two of them another conversation may have run.
func (c *Context) use() { c.engine.UseSlot(c.slot) }

// Pos is the position the next token would be fed at.
func (c *Context) Pos() int { return len(c.held) }

// Logits reads the distribution one hidden state produced. It needs no slot:
// the head is the same weights whichever conversation drew the state.
func (c *Context) Logits(hidden, out []float32) { c.engine.Logits(hidden, out) }

// Prefill brings the cache up to ids and returns the hidden state of the last
// position, with the number of positions it had to feed.
func (c *Context) Prefill(ids []int32) ([]float32, int, error) {
	if len(ids) == 0 {
		return nil, 0, fmt.Errorf("serve: an empty prompt")
	}
	if len(ids) > c.maxContext {
		return nil, 0, fmt.Errorf("serve: the conversation is %d positions and the context is %d: start the server with a larger -context, or send less", len(ids), c.maxContext)
	}
	c.use()
	c.expire()

	shared := 0
	for shared < len(c.held) && shared < len(ids) && c.held[shared] == ids[shared] {
		shared++
	}
	// The hidden state of a cached position was not kept, so the last position
	// of the prompt is fed whatever is shared.
	from := shared
	if from >= len(ids) {
		from = len(ids) - 1
	}
	// A rewind past what was written corrupts the ring of a window block.
	if len(c.held) > from && c.window > 0 {
		if early := from - c.window + 1; early > 0 {
			from = early
		} else {
			from = 0
		}
	}

	var hidden []float32
	for at := from; at < len(ids); at += promptBatch {
		to := min(at+promptBatch, len(ids))
		states := c.engine.ForwardBatch(ids[at:to], at)
		hidden = states[len(states)-1]
	}
	c.held = append(c.held[:0], ids...)
	c.last = c.now()
	return hidden, len(ids) - from, nil
}

// Advance feeds one drawn token and returns the state it produced.
func (c *Context) Advance(id int32) []float32 {
	c.use()
	states := c.engine.ForwardBatch([]int32{id}, len(c.held))
	c.held = append(c.held, id)
	c.last = c.now()
	return states[0]
}

// Full reports whether the context has no room left for another token.
func (c *Context) Full() bool { return len(c.held) >= c.maxContext }

// expire drops what is held once the time to live has passed. The memory is
// allocated at startup and is not released here: what expires is the record of
// whose conversation is in it.
func (c *Context) expire() {
	if c.ttl <= 0 || c.held == nil {
		return
	}
	if c.now().Sub(c.last) < c.ttl {
		return
	}
	c.engine.Reset()
	c.held = nil
}
