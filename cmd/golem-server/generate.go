package main

// Drawing one answer.
//
// Prose reaches the client as it is drawn. A tool call does not: as soon as
// a call opens the output is held back, and the call leaves in one
// piece once it has closed. A client that received half a call would hold half
// a function's arguments with no way to know it.
//
// What the model thinks before it answers is drawn like prose and leaves
// apart from it, as reasoning. The markers around it never leave: what could
// still be the beginning of one is held back until the next token says.

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
	Reasoning string // what the model thought before it answered
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

// Width is how many identifiers this generator scores, which is the width of
// the vocabulary a grammar has to read.
func (g *Generator) Width() int { return len(g.logits) }

// WithMaxTokens is the same generator over the same context, stopping sooner.
// A request naming its own limit gets one of these rather than changing the
// server's.
func (g *Generator) WithMaxTokens(n int) *Generator {
	out := *g
	out.maxTokens = n
	out.logits = make([]float32, len(g.logits))
	return &out
}

// endsInThought says whether the prompt's last words open a thought, blank
// lines aside. A few tokens are enough to hold the marker.
func (g *Generator) endsInThought(ids []int32, open string) bool {
	var tail strings.Builder
	for _, id := range ids[max(0, len(ids)-8):] {
		tail.WriteString(g.vocab.Piece(id, false))
	}
	return strings.HasSuffix(strings.TrimRight(tail.String(), " \n"), open)
}

// Delta is what one step of the drawing adds for the client: prose or
// reasoning, never both.
type Delta struct {
	Content   string
	Reasoning string
}

// Generate draws an answer for a prompt already rendered and encoded. emit,
// when it is not nil, receives each piece of prose and of reasoning as it is
// drawn.
//
// The context is the client's: when it hangs up, the drawing stops. A server
// with one model and one lock cannot afford to finish an answer nobody is
// waiting for — the next request is queued behind it.
func (g *Generator) Generate(ctx context.Context, ids []int32, p sample.Params, stop []string, emit func(Delta) error) (Answer, error) {
	return g.GeneratePrompt(ctx, engine.TextPrompt(ids), p, stop, emit)
}

// GeneratePrompt is the same for a prompt that may hold pictures.
func (g *Generator) GeneratePrompt(ctx context.Context, prompt engine.Prompt, p sample.Params, stop []string, emit func(Delta) error) (Answer, error) {
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
	// The penalties read the conversation, not only what is drawn from here
	// on: llama-server seeds its sampler with every prompt token before the
	// first draw (tools/server/server-context.cpp:254-260).
	sampler.Seed(prompt.Tokens())

	var drawn strings.Builder // everything drawn, calls and reasoning included
	open, close := g.tpl.ReasoningMarkers()
	sorted := sorter{open: open, close: close, call: g.tpl.CallOpen(), emit: emit}
	// A template that turns thinking on ends the prompt inside an open
	// thought, and the model writes only its close.
	if open != "" && g.endsInThought(prompt.Tokens(), open) {
		sorted.thinking, sorted.trim = true, true
	}

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
		return sorted.read(drawn.String(), false)
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
		if span := g.ctx.DraftSpan(); g.ctx.CanDraft(state) && answer.Generated+span-1 < g.maxTokens && g.ctx.Room(span) {
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
		// What came after the stop string is not part of the answer, though
		// what was already sent of it cannot be taken back.
		if err := sorted.read(*cut, true); err != nil {
			return answer, err
		}
		answer.Decode = time.Since(start)
		return g.finish(answer, &sorted)
	}
	if answer.Generated >= g.maxTokens {
		answer.Reason = "length"
	}
	if err := sorted.read(drawn.String(), true); err != nil {
		return answer, err
	}
	answer.Decode = time.Since(start)
	return g.finish(answer, &sorted)
}

// finish splits the prose into what came before the calls and the calls, and
// names the calls.
func (g *Generator) finish(answer Answer, sorted *sorter) (Answer, error) {
	answer.Reasoning = sorted.reasoning.String()
	before, calls, err := g.tpl.ParseCalls(sorted.content.String())
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

// sorter reads what is drawn into prose and reasoning as it arrives, and
// hands each on to emit once nothing drawn later can change what it is.
type sorter struct {
	open, close string // the reasoning's markers, empty for a template without
	call        string // where a call begins
	emit        func(Delta) error

	content, reasoning strings.Builder
	done               int  // how much of the drawn text is sorted
	thinking           bool // inside the reasoning
	calling            bool // past the first call: the rest is held, and parsed at the end
	trim               bool // the newlines here belong to a marker, not to the text
}

// read sorts the drawn text past what is already sorted. final says nothing
// more will be drawn, so nothing is held back any longer.
func (s *sorter) read(drawn string, final bool) error {
	for s.done < len(drawn) {
		rest := drawn[s.done:]
		switch {
		case s.calling:
			s.content.WriteString(rest)
			s.done = len(drawn)
		case s.trim:
			text := strings.TrimLeft(rest, "\n")
			s.done += len(rest) - len(text)
			s.trim = text == ""
		case s.thinking:
			if at := strings.Index(rest, s.close); at >= 0 {
				if err := s.think(strings.TrimRight(rest[:at], "\n")); err != nil {
					return err
				}
				s.done += at + len(s.close)
				s.thinking, s.trim = false, true
				continue
			}
			text := strings.TrimRight(rest, "\n")
			if !final {
				// Newlines are held with a marker's beginning: they end the
				// reasoning if the marker follows.
				text = strings.TrimRight(rest[:len(rest)-partial(rest, s.close)], "\n")
			}
			if err := s.think(text); err != nil {
				return err
			}
			s.done += len(text)
			if final {
				s.done = len(drawn)
			}
			return nil
		default:
			at, call := first(rest, s.open, s.call)
			if at < 0 {
				text := rest
				if !final {
					text = rest[:len(rest)-max(partial(rest, s.open), partial(rest, s.call))]
				}
				s.done += len(text)
				return s.say(text)
			}
			if err := s.say(rest[:at]); err != nil {
				return err
			}
			if call {
				s.content.WriteString(rest[at:])
				s.done, s.calling = len(drawn), true
				return nil
			}
			s.done += at + len(s.open)
			s.thinking, s.trim = true, true
		}
	}
	return nil
}

func (s *sorter) say(text string) error {
	if text == "" {
		return nil
	}
	s.content.WriteString(text)
	if s.emit == nil {
		return nil
	}
	return s.emit(Delta{Content: text})
}

func (s *sorter) think(text string) error {
	if text == "" {
		return nil
	}
	s.reasoning.WriteString(text)
	if s.emit == nil {
		return nil
	}
	return s.emit(Delta{Reasoning: text})
}

// first finds the earlier of a reasoning marker and a call in text, and says
// which it found. An empty marker is never found.
func first(text, open, call string) (at int, isCall bool) {
	at = -1
	if open != "" {
		at = strings.Index(text, open)
	}
	if call != "" {
		if c := strings.Index(text, call); c >= 0 && (at < 0 || c < at) {
			return c, true
		}
	}
	return at, false
}

// partial is how much of the end of text could be the beginning of marker.
func partial(text, marker string) int {
	for n := min(len(marker)-1, len(text)); n > 0; n-- {
		if strings.HasSuffix(text, marker[:n]) {
			return n
		}
	}
	return 0
}
