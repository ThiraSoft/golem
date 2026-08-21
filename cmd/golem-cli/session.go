package main

// One conversation with the model.
//
// Every turn re-renders the whole conversation, because the template is the
// only thing that knows how to write one and no two checkpoints write it the
// same way. What keeps that cheap is that the cache is not rebuilt with it:
// the new render is encoded, compared against the tokens the cache already
// holds, and only the tail that differs is fed. A turn therefore costs a turn,
// not a conversation — which is the whole point, since re-reading three
// gigabytes of weights for every token already spoken is what this avoids.

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/sample"
)

// forward is the part of a model a conversation uses. Reset is not in it: a
// conversation never resets, cmd/golem-cli builds a new one instead. Named here so that the
// loop can be tested without three gigabytes of weights.
type forward interface {
	ForwardBatch(tokens []int32, startPos int) [][]float32
	Logits(hidden, out []float32)
}

// promptBatch is how many positions of a prompt go through the model together.
// A batch reads each matrix of weights once for all of it, which is what makes
// reading a prompt several times faster than speaking one.
//
// Thirty-two is measured, not chosen: the gain is already flat at sixteen —
// past that the weights are no longer what limits the pass — and beyond
// sixty-four it turns back down, because the activations of a batch stop
// fitting in the caches and every position attends to every position before it.
const promptBatch = 32

// vocabulary is the part of bpe.Vocab a conversation uses.
type vocabulary interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Piece(id int32, special bool) string
	IsEOG(id int32) bool
}

type Session struct {
	model      forward
	vocab      vocabulary
	tpl        chat.Template
	sampler    *sample.Sampler
	vocabSize  int
	maxContext int
	maxTokens  int
	thinking   bool

	// history is the conversation as messages, because the template is what
	// turns it into text and only the template knows how.
	history []chat.Message
	// held is what the cache holds, position by position. A turn renders the
	// whole conversation and feeds only what is not already a prefix of this.
	held   []int32
	logits []float32
}

// Turn is what one exchange cost.
type Turn struct {
	Prompt    int // tokens fed
	Generated int // tokens drawn
	Prefill   time.Duration
	Decode    time.Duration
	Text      string
	Truncated bool // stopped on a limit rather than on an end-of-turn token
}

func NewSession(m forward, v vocabulary, tpl chat.Template, p sample.Params,
	vocabSize, maxContext, maxTokens int, system string, thinking bool) *Session {
	s := &Session{
		model:      m,
		vocab:      v,
		tpl:        tpl,
		sampler:    sample.New(p),
		vocabSize:  vocabSize,
		maxContext: maxContext,
		maxTokens:  maxTokens,
		thinking:   thinking,
		logits:     make([]float32, vocabSize),
	}
	if system != "" {
		s.history = append(s.history, chat.Message{Role: "system", Content: system})
	}
	return s
}

// Ask feeds one user message and writes the answer to w as it comes.
func (s *Session) Ask(text string, w io.Writer) (Turn, error) {
	s.history = append(s.history, chat.Message{Role: "user", Content: strings.TrimSpace(text)})
	rendered, err := s.tpl.Render(s.history, chat.Options{
		EnableThinking:      s.thinking,
		AddGenerationPrompt: true,
	})
	if err != nil {
		return Turn{}, err
	}
	ids := s.vocab.Encode(rendered, false, true)
	if len(ids) >= s.maxContext {
		return Turn{}, fmt.Errorf("the conversation no longer fits in %d positions: pass a larger -context, or start again", s.maxContext)
	}

	// What the cache holds is a prefix of what this turn needs, unless the
	// template rewrote something behind us — a past answer losing its thinking
	// block, for instance. Either way, feeding starts where the two stop
	// agreeing, and the last position is always fed because its hidden state
	// was not kept.
	shared := 0
	for shared < len(s.held) && shared < len(ids) && s.held[shared] == ids[shared] {
		shared++
	}
	from := shared
	if from >= len(ids) {
		from = len(ids) - 1
	}

	start := time.Now()
	var hidden []float32
	for at := from; at < len(ids); at += promptBatch {
		to := min(at+promptBatch, len(ids))
		states := s.model.ForwardBatch(ids[at:to], at)
		hidden = states[len(states)-1]
	}
	s.held = append(s.held[:0], ids...)
	turn := Turn{Prompt: len(ids) - from, Prefill: time.Since(start)}

	var answer strings.Builder
	start = time.Now()
	last := int32(-1)
	for turn.Generated < s.maxTokens && len(s.held) < s.maxContext {
		s.model.Logits(hidden, s.logits)
		id := s.sampler.Pick(s.logits)
		turn.Generated++
		last = id
		if s.vocab.IsEOG(id) {
			break
		}
		piece := s.vocab.Piece(id, false)
		answer.WriteString(piece)
		if w != nil {
			if _, err := io.WriteString(w, piece); err != nil {
				return turn, err
			}
		}
		hidden = s.model.ForwardBatch([]int32{id}, len(s.held))[0]
		s.held = append(s.held, id)
	}
	turn.Decode = time.Since(start)
	turn.Text = answer.String()
	turn.Truncated = last < 0 || !s.vocab.IsEOG(last)

	// The token that ended the turn was drawn but never fed: feeding it would
	// cost a forward pass for a marker the next render is about to carry
	// anyway, and the prefix comparison will put it in at the right position.
	s.history = append(s.history, chat.Message{Role: "assistant", Content: turn.Text})
	return turn, nil
}
