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

	"github.com/ThiraSoft/golem/gemma"
)

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
	runner     *Runner
	slot       int // which of the model's caches this context is
	window     int // the largest sliding window; 0 when every block is global
	maxContext int
	now        func() time.Time
	ttl        time.Duration

	held []int32 // the tokens the cache holds, position by position
	last time.Time
}

func NewContext(r *Runner, window, maxContext int, now func() time.Time, ttl time.Duration) *Context {
	return &Context{runner: r, window: window, maxContext: maxContext, now: now, ttl: ttl}
}

// NewSlotContext is the same, for one of several caches the model holds.
func NewSlotContext(r *Runner, slot, window, maxContext int, now func() time.Time, ttl time.Duration) *Context {
	c := NewContext(r, window, maxContext, now, ttl)
	c.slot = slot
	return c
}

// Pos is the position the next token would be fed at.
func (c *Context) Pos() int { return len(c.held) }

// Prefill brings the cache up to ids, scores the last position into logits,
// and returns how many positions it had to feed.
func (c *Context) Prefill(ids []int32, logits []float32) (int, error) {
	return c.PrefillPrompt(&gemma.Prompt{Tokens: ids}, logits)
}

// PrefillPrompt is the same for a prompt that may hold pictures: the rows go
// in where the soft tokens are, and a batch is never cut inside one.
func (c *Context) PrefillPrompt(p *gemma.Prompt, logits []float32) (int, error) {
	ids := p.Tokens
	if len(ids) == 0 {
		return 0, fmt.Errorf("serve: an empty prompt")
	}
	if len(ids) > c.maxContext {
		return 0, fmt.Errorf("serve: the conversation is %d positions and the context is %d: start the server with a larger -context, or send less", len(ids), c.maxContext)
	}
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

	for at := from; at < len(ids); {
		// A batch may not be cut inside a picture: every key of a span has to
		// be in the cache before any of its queries is scored, which holds
		// within one pass and not across two.
		to := p.Boundary(at, at+promptBatch)
		// Only the chunk that ends the prompt is scored: the ones before it
		// are read for their keys and values alone.
		var out []float32
		if to == len(ids) {
			out = logits
		}
		if len(p.Spans) == 0 {
			c.runner.Forward(c.slot, ids[at:to], span(at, to-at), out)
		} else {
			chunk := p.Slice(at, to)
			c.runner.ForwardEmbedded(c.slot, chunk.Tokens, chunk.Embeds, chunk.PLE,
				span(at, to-at), chunk.Until(at), out)
		}
		at = to
	}
	c.held = append(c.held[:0], ids...)
	c.last = c.now()
	return len(ids) - from, nil
}

// Advance feeds one drawn token and scores what it produced.
func (c *Context) Advance(id int32, logits []float32) {
	c.runner.Forward(c.slot, []int32{id}, span(len(c.held), 1), logits)
	c.held = append(c.held, id)
	c.last = c.now()
}

// span is the positions a run of n tokens starting at from occupies.
func span(from, n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = from + i
	}
	return out
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
	c.runner.Reset(c.slot)
	c.held = nil
}
