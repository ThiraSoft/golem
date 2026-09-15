package main

// What the KV cache holds, across requests that carry no state.
//
// /v1/chat/completions is stateless: every request sends the whole
// conversation. The cache is not — it is the whole reason a turn costs one turn
// rather than the conversation. So the server remembers which tokens are in the
// cache, and a request only pays for what its prompt does not share with them.
//
// One rule earns its own paragraph. A sliding-window block stores its keys in a
// ring of C slots, position p in slot p mod C, and C is the window and one pass
// more (gemma/cache.go says why). A conversation that wrote up to Q and is
// rewound to a shared prefix P has overwritten the slots of positions Q-C and
// below, and some of them can still be inside the window of P. Resuming at P
// would read the keys of the other conversation in their place. So the context
// remembers which position each slot last received, and a prefill starts at
// the first position whose window is intact, feeding again everything after
// it. Rewriting positions is idempotent for the global blocks. Appending,
// which is what a conversation growing by one exchange does, overwrites
// nothing still visible and costs nothing.
//
// This used to restart a window early, P-W+1, reasoning on a ring of exactly
// the window. Once the ring grew by a pass that stopped being enough: a
// village whose characters share a long lore and take turns on a slot read the
// keys of the previous character, and answered as someone else.

import (
	"fmt"
	"time"

	"github.com/ThiraSoft/golem/engine"
)

// Vocabulary is the part of engine.Vocabulary the server uses — Gemma's
// tokenizer or Qwen's, whichever the engine loaded.
type Vocabulary interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Piece(id int32, special bool) string
	IsEOG(id int32) bool
}

// promptBatch is how many positions go through the model together on the
// processor. Thirty-two is what cmd/golem-cli measured: the gain is flat by
// sixteen, and past sixty-four the activations stop fitting in the caches.
//
// A card has no such ceiling and a far higher floor: it is idle at thirty-two.
// devicePassWidth is what a model whose blocks are on one reads instead, and
// Runner.PassWidth is which of the two this server is using — main.go asks the
// model where its blocks are and says so once.
const promptBatch = 32

// devicePassWidth is the same for a model on a card. Measured on a Gemma 4 26B
// A4B and an RX 9070 XT, one conversation, positions a second by the width of
// the pass: 950 at 32, 2044 at 64, 3797 at 128, 4981 at 256, 5486 at 512.
//
// Five hundred and twelve, which is what vk's stack carries at most. It is the
// widest pass and also the longest — 93ms, against 51ms at 256 — so a
// conversation waiting on its next token waits that much longer for one that
// is reading a prompt. Forty milliseconds bought at the price of a tenth of
// the prompt rate is not a trade worth making: the wait is a fraction of a
// prompt, once, and the rate is every prompt.
const devicePassWidth = 512

type Context struct {
	runner *Runner
	slot   int // which of the model's caches this context is
	window int // the largest sliding window; 0 when every block is global
	ring   int // slots in that window's ring; the window itself when unset

	// owner is, for each slot of the window ring, the position it last
	// received, or -1. A slot whose owner is not the position read from it
	// holds some other position's keys.
	owner      []int
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

// SetRing gives the size of the window ring, which main.go reads from the
// model. Without it the ring is taken to be the window.
func (c *Context) SetRing(n int) { c.ring = n; c.owner = nil }

// wrote records that positions from … from+n-1 have been written to the ring.
func (c *Context) wrote(from, n int) {
	if c.window == 0 {
		return
	}
	ring := c.ringSize()
	if c.owner == nil {
		c.owner = make([]int, ring)
		for i := range c.owner {
			c.owner[i] = -1
		}
	}
	for p := from; p < from+n; p++ {
		c.owner[p%ring] = p
	}
}

func (c *Context) ringSize() int {
	if c.ring > 0 {
		return c.ring
	}
	return c.window
}

// intactFrom is the latest position at or before from where a prefill can
// resume: every position its window sees still sits in its own slot. A
// position found overwritten is fed again, so the search moves to it and
// checks its window in turn.
func (c *Context) intactFrom(from int) int {
	if c.window == 0 {
		return from
	}
	ring := c.ringSize()
	for from > 0 {
		bad := -1
		for p := max(0, from-c.window+1); p < from; p++ {
			if c.owner == nil || c.owner[p%ring] != p {
				bad = p
				break
			}
		}
		if bad < 0 {
			return from
		}
		from = bad
	}
	return 0
}

// Pos is the position the next token would be fed at.
func (c *Context) Pos() int { return len(c.held) }

// Prefill brings the cache up to ids, scores the last position into logits,
// and returns how many positions it had to feed.
func (c *Context) Prefill(ids []int32, logits []float32) (int, error) {
	return c.PrefillPrompt(engine.TextPrompt(ids), logits)
}

// PrefillPrompt is the same for a prompt that may hold pictures: the rows go
// in where the soft tokens are, and a batch is never cut inside one.
func (c *Context) PrefillPrompt(p engine.Prompt, logits []float32) (int, error) {
	return c.PrefillPromptState(p, logits, nil)
}

// PrefillPromptState is PrefillPrompt that also keeps the hidden state of the
// last position. A conversation that is about to draft needs it: the prediction
// block reads the state of the token before the one it drafts from.
func (c *Context) PrefillPromptState(p engine.Prompt, logits []float32, state *[]float32) (int, error) {
	ids := p.Tokens()
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
	// A window block's ring may hold another conversation's keys where this
	// one's window still looks.
	from = c.intactFrom(from)

	for at := from; at < len(ids); {
		// A batch may not be cut inside a picture: every key of a span has to
		// be in the cache before any of its queries is scored, which holds
		// within one pass and not across two.
		to := p.Boundary(at, at+c.runner.PassWidth())
		// Only the chunk that ends the prompt is scored: the ones before it
		// are read for their keys and values alone.
		var out []float32
		var keep *[]float32
		if to == len(ids) {
			out, keep = logits, state
		}
		if p.Embeds() == nil {
			c.runner.ForwardState(c.slot, ids[at:to], span(at, to-at), out, keep)
		} else {
			chunk := p.Slice(at, to)
			ple, until, axes := chunk.Extras(at)
			c.runner.ForwardEmbedded(c.slot, chunk.Tokens(), chunk.Embeds(), ple,
				span(at, to-at), until, axes, out)
			if keep != nil {
				// A prompt carrying a picture goes through the vision path,
				// which does not keep states. Such a conversation draws a
				// token at a time until its next plain pass.
				*keep = (*keep)[:0]
			}
		}
		c.wrote(at, to-at)
		at = to
	}
	c.held = append(c.held[:0], ids...)
	c.last = c.now()
	return len(ids) - from, nil
}

// Advance feeds one drawn token and scores what it produced.
func (c *Context) Advance(id int32, logits []float32) {
	c.AdvanceState(id, logits, nil)
}

// AdvanceState is Advance that also keeps the hidden state the token produced.
func (c *Context) AdvanceState(id int32, logits []float32, state *[]float32) {
	c.runner.ForwardState(c.slot, []int32{id}, span(len(c.held), 1), logits, state)
	c.wrote(len(c.held), 1)
	c.held = append(c.held, id)
	c.last = c.now()
}

// CanDraft reports whether this conversation should draft its next token: the
// model has to carry a prediction block, the runner has to be otherwise idle,
// and there has to be a state to draft from — a prompt that ended in a picture
// leaves none.
func (c *Context) CanDraft(state []float32) bool {
	return len(state) > 0 && c.runner.CanDraft()
}

// DraftSpan is the most positions a drafting step writes: the room to leave.
func (c *Context) DraftSpan() int { return c.runner.DraftSpan() }

// Draft feeds the token just drawn and draws whatever the prediction block got
// right after it, in one reading of the weights.
//
// It returns the tokens decided after id. The last of them has been drawn but
// not fed — it is the caller's next id — and everything before it is in the
// cache. state is replaced by the hidden state of the last of them.
func (c *Context) Draft(id int32, state *[]float32, pick func([]float32) int32) ([]int32, error) {
	next, h, err := c.runner.Draft(c.slot, id, *state, len(c.held), pick)
	// The refused guesses were written too, at positions past what is held.
	c.wrote(len(c.held), c.runner.DraftSpan())
	if err != nil {
		return nil, err
	}
	c.held = append(c.held, id)
	c.held = append(c.held, next[:len(next)-1]...)
	*state = append((*state)[:0], h...)
	c.last = c.now()
	return next, nil
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
func (c *Context) Full() bool { return !c.Room(1) }

// Room reports whether n more positions fit, which a speculative step asks
// before it writes two.
func (c *Context) Room(n int) bool { return len(c.held)+n <= c.maxContext }

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
	c.owner = nil
}
