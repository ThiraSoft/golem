package main

// Drawing one answer.
//
// Prose reaches the client as it is drawn. A tool call does not: as soon as
// a call opens the output is held back, and the call leaves in one
// piece once it has closed. A client that received half a call would hold half
// a function's arguments with no way to know it.

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/engine"
	"github.com/ThiraSoft/golem/sample"
)

type Generator struct {
	ctx       *Context
	vocab     Vocabulary
	tpl       chat.Template
	logits    []float32
	maxTokens int
	calls     *int // how many calls this server has handed out, for identifiers
}

// Answer is one answer and what it cost.
type Answer struct {
	Text      string
	ToolCalls []chat.ToolCall
	Prompt    int // positions actually fed, which the cache's prefix makes small
	Generated int
	Prefill   time.Duration // reading the prompt
	Decode    time.Duration // drawing the answer
	Reason    string        // "stop", "length" or "tool_calls"
}

func NewGenerator(ctx *Context, v Vocabulary, tpl chat.Template, vocabSize, maxTokens int) *Generator {
	calls := 0
	return &Generator{ctx: ctx, vocab: v, tpl: tpl, logits: make([]float32, vocabSize),
		maxTokens: maxTokens, calls: &calls}
}

// WithMaxTokens is the same generator over the same context, stopping sooner.
// A request naming its own limit gets one of these rather than changing the
// server's.
func (g *Generator) WithMaxTokens(n int) *Generator {
	out := *g
	out.maxTokens = n
	out.logits = make([]float32, len(g.logits))
	return &out
}

// Generate draws an answer for a prompt already rendered and encoded. emit,
// when it is not nil, receives each piece of prose as it is drawn.
//
// The context is the client's: when it hangs up, the drawing stops. A server
// with one model and one lock cannot afford to finish an answer nobody is
// waiting for — the next request is queued behind it.
func (g *Generator) Generate(ctx context.Context, ids []int32, p sample.Params, stop []string, emit func(string) error) (Answer, error) {
	return g.GeneratePrompt(ctx, engine.TextPrompt(ids), p, stop, emit)
}

// GeneratePrompt is the same for a prompt that may hold pictures.
func (g *Generator) GeneratePrompt(ctx context.Context, prompt engine.Prompt, p sample.Params, stop []string, emit func(string) error) (Answer, error) {
	start := time.Now()
	// The state the prompt ended on, which is the prediction block's second
	// input. A conversation that cannot draft never reads it.
	var state []float32
	fed, err := g.ctx.PrefillPromptState(prompt, g.logits, &state)
	if err != nil {
		return Answer{}, err
	}
	answer := Answer{Prompt: fed, Prefill: time.Since(start), Reason: "stop"}
	start = time.Now()
	sampler := sample.New(p)

	var drawn strings.Builder // everything drawn, calls included
	sent := 0                 // how much of it has left through emit
	inCall := false

	// take puts one drawn token into the answer and says whether the answer
	// ends there. A stop string is the one ending that decides what the answer
	// holds, so it comes back with the text rather than as a flag.
	stopped := false
	var cut *string
	take := func(id int32) error {
		answer.Generated++
		if g.vocab.IsEOG(id) {
			stopped = true
			return nil
		}
		piece := g.vocab.Piece(id, false)
		if text, hit := cutAtStop(drawn.String()+piece, stop); hit {
			stopped, cut = true, &text
			return nil
		}
		drawn.WriteString(piece)

		// Prose goes out up to the first call; from there on the output is
		// held, and what it holds leaves as a call rather than as text.
		if inCall {
			return nil
		}
		text := drawn.String()[sent:]
		if at := strings.Index(text, g.tpl.CallOpen()); at >= 0 {
			text, inCall = text[:at], true
		}
		if text != "" && emit != nil {
			if err := emit(text); err != nil {
				return err
			}
		}
		sent += len(text)
		return nil
	}

	// pending is a token a speculative step already drew from the model's own
	// distribution. Drawing it again would be a second reading of the head,
	// which is the largest matrix in the model.
	pending := int32(-1)

	for answer.Generated < g.maxTokens && !g.ctx.Full() {
		if err := ctx.Err(); err != nil {
			return answer, err
		}
		id := pending
		if id < 0 {
			id = sampler.Pick(g.logits)
		}
		pending = -1
		if err := take(id); err != nil {
			return answer, err
		}
		if stopped {
			break
		}

		// Two tokens out of one reading of the weights, when the checkpoint
		// carries a prediction block and no other conversation is waiting for
		// a pass — a pass carrying two of them is the better bargain, and
		// Runner.CanDraft is what weighs the two.
		if g.ctx.CanDraft(state) && answer.Generated+1 < g.maxTokens && g.ctx.Room(2) {
			next, err := g.ctx.Draft(id, &state, sampler.Pick)
			if err != nil {
				return answer, err
			}
			// The last of what came back is the token after everything the
			// model has read, which is this loop's next id. Everything before
			// it is decided and is already in the cache.
			for _, tok := range next[:len(next)-1] {
				if err := take(tok); err != nil {
					return answer, err
				}
				if stopped {
					break
				}
			}
			if stopped {
				break
			}
			pending = next[len(next)-1]
			continue
		}

		g.ctx.AdvanceState(id, g.logits, &state)
	}
	if cut != nil {
		answer.Decode = time.Since(start)
		return g.finish(answer, *cut)
	}
	if answer.Generated >= g.maxTokens {
		answer.Reason = "length"
	}
	answer.Decode = time.Since(start)
	return g.finish(answer, drawn.String())
}

// finish splits what was drawn into prose and calls, and names the calls.
func (g *Generator) finish(answer Answer, drawn string) (Answer, error) {
	before, calls, err := g.tpl.ParseCalls(drawn)
	if err != nil {
		return answer, fmt.Errorf("the model wrote a call this server cannot read: %w", err)
	}
	answer.Text = before
	for i := range calls {
		*g.calls++
		calls[i].ID = "call_" + strconv.Itoa(*g.calls)
	}
	answer.ToolCalls = calls
	if len(calls) > 0 {
		answer.Reason = "tool_calls"
	}
	return answer, nil
}

// cutAtStop reports whether one of the stop strings has appeared, and returns
// what came before it.
func cutAtStop(text string, stop []string) (string, bool) {
	for _, s := range stop {
		if s == "" {
			continue
		}
		if i := strings.Index(text, s); i >= 0 {
			return text[:i], true
		}
	}
	return text, false
}
