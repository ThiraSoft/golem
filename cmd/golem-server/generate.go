package main

// Drawing one answer.
//
// Prose reaches the client as it is drawn. A tool call does not: as soon as
// <|tool_call> appears the output is held back, and the call leaves in one
// piece once it has closed. A client that received half a call would hold half
// a function's arguments with no way to know it.

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/sample"
)

// callOpen is where prose stops and a call begins.
const callOpen = "<|tool_call>"

type Generator struct {
	ctx       *Context
	vocab     Vocabulary
	logits    []float32
	maxTokens int
	calls     *int // how many calls this server has handed out, for identifiers
}

// Answer is one answer and what it cost.
type Answer struct {
	Text      string
	ToolCalls []gemma.ToolCall
	Prompt    int // positions actually fed, which the cache's prefix makes small
	Generated int
	Prefill   time.Duration // reading the prompt
	Decode    time.Duration // drawing the answer
	Reason    string        // "stop", "length" or "tool_calls"
}

func NewGenerator(ctx *Context, v Vocabulary, vocabSize, maxTokens int) *Generator {
	calls := 0
	return &Generator{ctx: ctx, vocab: v, logits: make([]float32, vocabSize),
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
	start := time.Now()
	hidden, fed, err := g.ctx.Prefill(ids)
	if err != nil {
		return Answer{}, err
	}
	answer := Answer{Prompt: fed, Prefill: time.Since(start), Reason: "stop"}
	start = time.Now()
	sampler := sample.New(p)

	var drawn strings.Builder // everything drawn, calls included
	sent := 0                 // how much of it has left through emit
	inCall := false

	for answer.Generated < g.maxTokens && !g.ctx.Full() {
		if err := ctx.Err(); err != nil {
			return answer, err
		}
		g.ctx.Logits(hidden, g.logits)
		id := sampler.Pick(g.logits)
		answer.Generated++
		if g.vocab.IsEOG(id) {
			break
		}
		piece := g.vocab.Piece(id, false)

		if cut, hit := cutAtStop(drawn.String()+piece, stop); hit {
			answer.Decode = time.Since(start)
			return g.finish(answer, cut)
		}
		drawn.WriteString(piece)

		// Prose goes out up to the first call; from there on the output is
		// held, and what it holds leaves as a call rather than as text.
		if !inCall {
			text := drawn.String()[sent:]
			if at := strings.Index(text, callOpen); at >= 0 {
				text, inCall = text[:at], true
			}
			if text != "" && emit != nil {
				if err := emit(text); err != nil {
					return answer, err
				}
			}
			sent += len(text)
		}
		hidden = g.ctx.Advance(id)
	}
	if answer.Generated >= g.maxTokens {
		answer.Reason = "length"
	}
	answer.Decode = time.Since(start)
	return g.finish(answer, drawn.String())
}

// finish splits what was drawn into prose and calls, and names the calls.
func (g *Generator) finish(answer Answer, drawn string) (Answer, error) {
	before, calls, err := gemma.ParseToolCalls(drawn)
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
