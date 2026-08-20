package main

// One conversation with the model.
//
// The context is built once and never rebuilt: the template's output grows by
// appending, so each turn only has to encode what is new — the closing of the
// model's last turn, the user's message, and the header of the next answer.
// Re-rendering the whole conversation every turn would re-read three gigabytes
// of weights for every token already spoken.

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/sample"
)

// engine is the part of gemma.Model a conversation uses. Named here so that the
// loop can be tested without three gigabytes of weights.
type engine interface {
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
	model      engine
	vocab      vocabulary
	sampler    *sample.Sampler
	vocabSize  int
	maxContext int
	maxTokens  int
	system     string
	thinking   bool

	pos     int
	started bool
	// pending closes the turn the model has just spoken. The end-of-turn token
	// is drawn but never fed — feeding it would cost a forward pass for a
	// marker the next prompt is about to carry anyway — so the closing belongs
	// to the front of the next encoding.
	pending string
	logits  []float32
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

func NewSession(m engine, v vocabulary, p sample.Params, vocabSize, maxContext, maxTokens int, system string, thinking bool) *Session {
	return &Session{
		model:      m,
		vocab:      v,
		sampler:    sample.New(p),
		vocabSize:  vocabSize,
		maxContext: maxContext,
		maxTokens:  maxTokens,
		system:     system,
		thinking:   thinking,
		logits:     make([]float32, vocabSize),
	}
}

// prompt is the text to encode for a user message: the whole template on the
// first turn, its continuation afterwards.
func (s *Session) prompt(text string) (string, error) {
	text = strings.TrimSpace(text)
	if !s.started {
		var messages []gemma.Message
		if s.system != "" {
			messages = append(messages, gemma.Message{Role: "system", Content: s.system})
		}
		messages = append(messages, gemma.Message{Role: "user", Content: text})
		return gemma.RenderChat(messages, gemma.ChatOptions{
			EnableThinking:      s.thinking,
			AddGenerationPrompt: true,
		})
	}
	return s.pending + "<|turn>user\n" + text + "<turn|>\n<|turn>model\n", nil
}

// Ask feeds one user message and writes the answer to w as it comes.
func (s *Session) Ask(text string, w io.Writer) (Turn, error) {
	prompt, err := s.prompt(text)
	if err != nil {
		return Turn{}, err
	}
	ids := s.vocab.Encode(prompt, false, true)
	if s.pos+len(ids) >= s.maxContext {
		return Turn{}, fmt.Errorf("the conversation no longer fits in %d positions: pass a larger -context, or start again", s.maxContext)
	}

	start := time.Now()
	var hidden []float32
	for from := 0; from < len(ids); from += promptBatch {
		to := min(from+promptBatch, len(ids))
		states := s.model.ForwardBatch(ids[from:to], s.pos)
		hidden = states[len(states)-1]
		s.pos += to - from
	}
	turn := Turn{Prompt: len(ids), Prefill: time.Since(start)}
	s.started = true

	var answer strings.Builder
	start = time.Now()
	last := int32(-1)
	for turn.Generated < s.maxTokens && s.pos < s.maxContext {
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
		hidden = s.model.ForwardBatch([]int32{id}, s.pos)[0]
		s.pos++
	}
	turn.Decode = time.Since(start)
	turn.Text = answer.String()
	turn.Truncated = last < 0 || !s.vocab.IsEOG(last)

	// The token that ended the turn was drawn but never fed, so the closing
	// marker is still missing from the context and the next turn opens with it.
	// The template writes <turn|> whatever the model stopped on.
	s.pending = "<turn|>\n"
	return turn, nil
}
