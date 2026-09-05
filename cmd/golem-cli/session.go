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
	"github.com/ThiraSoft/golem/engine"
	"github.com/ThiraSoft/golem/grammar"
	"github.com/ThiraSoft/golem/qwen35"
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
//
// That ceiling is the processor's caches and a card has none of it. Measured on
// a Gemma 4 26B A4B and an RX 9070 XT, positions a second by the width of the
// pass: 950 at 32, 2044 at 64, 3797 at 128, 4981 at 256, 5486 at 512. So a
// model whose blocks are on a card reads devicePassWidth instead, which is the
// widest the stack carries; Session.OnDevice is how it is told.
const promptBatch = 32

// devicePassWidth is that width, for a model on a card.
const devicePassWidth = 512

// speculative is the part of a model that can draft with a prediction block,
// and speculator is one prepared to. Nothing outside qwen35 implements either
// yet, and a model that does not simply generates a token at a time.
type speculative interface {
	Speculate() bool
	NewSpeculator() (*qwen35.Speculator, error)
}

type speculator interface {
	Step(token int32, hidden []float32, pos int, pick func([]float32) int32) ([]int32, []float32, error)
	Rate() (accepted, drafted int)
}

// vocabulary is the part of engine.Vocabulary a conversation uses — Gemma's
// tokenizer or Qwen's, whichever the engine loaded.
type vocabulary interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Piece(id int32, special bool) string
	IsEOG(id int32) bool
}

type Session struct {
	model forward
	// vision is nil unless a projector was opened. When it is not, a turn goes
	// through it whether or not this conversation has ever held a picture:
	// the tokens are the same either way, and one path is easier to trust
	// than two.
	vision     engine.Media
	vocab      vocabulary
	tpl        chat.Template
	sampler    *sample.Sampler
	vocabSize  int
	maxContext int
	maxTokens  int
	thinking   bool
	// noDraft turns the prediction block off for a checkpoint that carries
	// one. Drafting is what the block is for and is on wherever it can be, so
	// this exists to measure it: what a draft is worth is a number about a
	// model and a card, and it moves whenever the kernels do.
	noDraft bool

	// rules is the grammar every answer of this conversation has to stay
	// inside, and tokens is the vocabulary a grammar reads through. Both are
	// nil unless the command line asked for one.
	rules  *grammar.Rules
	tokens *grammar.Tokens

	// history is the conversation as messages, because the template is what
	// turns it into text and only the template knows how.
	history []chat.Message
	// held is what the cache holds, position by position. A turn renders the
	// whole conversation and feeds only what is not already a prefix of this.
	held   []int32
	logits []float32

	// width is how wide a prompt pass may be; zero means the processor's.
	width int
	// encoded is what the tower made of each picture, kept for as long as the
	// conversation holds it. A picture costs a second to look at and the
	// conversation is re-rendered every turn; looking again each time would
	// make the third turn about a picture cost what the first did.
	encoded [][][]float32
	// heard is the same for the recordings, kept for the same reason: a
	// conformer is slower than a vision tower, not faster.
	heard [][][]float32
}

// SetVision gives this conversation an encoder for the pictures its turns
// carry. Without one, a turn carrying a picture is refused rather than
// answered about nothing.
func (s *Session) SetVision(v engine.Media) { s.vision = v }

// Turn is what one exchange cost.
type Turn struct {
	Prompt    int // tokens fed
	Generated int // tokens drawn
	Prefill   time.Duration
	Decode    time.Duration
	Text      string
	Truncated bool // stopped on a limit rather than on an end-of-turn token
	// Accepted and Drafted are the prediction block's, and zero when nothing
	// drafted. A step that costs a pass of two and a reading of the head three
	// times is only worth taking when most drafts land, and which it is is a
	// property of the sampler as much as of the model: a draft is taken at the
	// peak, so a sampler that does not pick the peak disagrees with it by
	// construction. Reporting it is how that gets noticed.
	Accepted, Drafted int
}

// promptWidth is how many positions of a prompt go through the model together,
// which is the processor's promptBatch unless the blocks are on a card.
func (s *Session) promptWidth() int {
	if s.width == 0 {
		return promptBatch
	}
	return s.width
}

// OnDevice says the model's blocks are on a card, which is the only thing that
// changes the width. cmd/golem-cli calls it once, before the first prompt.
func (s *Session) OnDevice() { s.width = devicePassWidth }

// NoDraft makes this conversation generate a token at a time even where the
// checkpoint carries a prediction block. See Session.noDraft.
func (s *Session) NoDraft() { s.noDraft = true }

// Constrain makes every answer stay inside a grammar, written in GBNF. The
// vocabulary is read once here rather than once a turn: it is a quarter of a
// million pieces.
func (s *Session) Constrain(src string) error {
	rules, err := grammar.Parse(src)
	if err != nil {
		return err
	}
	s.rules = rules
	s.tokens = grammar.NewTokens(s.vocabSize,
		func(id int32) string { return s.vocab.Piece(id, false) }, s.vocab.IsEOG)
	return nil
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
	return s.AskWith(text, nil, w)
}

// AskWith is Ask with pictures attached to the turn, each still in the bytes
// some encoder wrote.
func (s *Session) AskWith(text string, images [][]byte, w io.Writer) (Turn, error) {
	return s.AskWithMedia(text, images, nil, w)
}

// AskWithMedia is Ask with pictures and recordings attached to the turn, each
// still in the bytes some encoder wrote.
func (s *Session) AskWithMedia(text string, images, audio [][]byte, w io.Writer) (Turn, error) {
	if len(images) > 0 && (s.vision == nil || !s.vision.CanSee()) {
		return Turn{}, fmt.Errorf("this model was opened without a projector that can see, so it cannot look at a picture: pass -mmproj")
	}
	if len(audio) > 0 && (s.vision == nil || !s.vision.CanHear()) {
		return Turn{}, fmt.Errorf("this model was opened without a projector that can hear, so it cannot listen: pass -mmproj")
	}
	for _, raw := range images {
		rows, err := s.vision.EncodeImage(raw)
		if err != nil {
			return Turn{}, err
		}
		s.encoded = append(s.encoded, rows)
	}
	for _, raw := range audio {
		rows, err := s.vision.EncodeAudio(raw)
		if err != nil {
			return Turn{}, err
		}
		s.heard = append(s.heard, rows)
	}
	s.history = append(s.history, chat.Message{
		Role: "user", Content: strings.TrimSpace(text), Images: images, Audio: audio,
	})
	rendered, err := s.tpl.Render(s.history, chat.Options{
		EnableThinking:      s.thinking,
		AddGenerationPrompt: true,
	})
	if err != nil {
		return Turn{}, err
	}
	ids := s.vocab.Encode(rendered, false, true)

	// The rendered conversation carries an empty pair of markers where each
	// picture goes; the soft tokens go between them, and what comes back is
	// the prompt the model actually reads.
	var prompt engine.Prompt
	if s.vision != nil {
		p, err := s.vision.Prompt(ids, s.encoded, s.heard)
		if err != nil {
			return Turn{}, err
		}
		prompt, ids = p, p.Tokens()
	}
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
	if prompt != nil {
		// A batch may not be cut inside a picture: every key of a span has to
		// be in the cache before any of its queries is scored, which holds
		// within one pass and not across two.
		for at := from; at < len(ids); {
			to := prompt.Boundary(at, at+s.promptWidth())
			states := s.vision.ForwardPrompt(prompt.Slice(at, to), at)
			hidden = states[len(states)-1]
			at = to
		}
	} else {
		width := s.promptWidth()
		for at := from; at < len(ids); at += width {
			to := min(at+width, len(ids))
			states := s.model.ForwardBatch(ids[at:to], at)
			hidden = states[len(states)-1]
		}
	}
	if s.rules != nil {
		// A grammar constrains an answer, not a conversation: each turn gets a
		// fresh walk from the root.
		s.sampler.Constrain(grammar.New(s.rules, s.tokens))
	}
	s.held = append(s.held[:0], ids...)
	// The penalties read the conversation and not only what is drawn: every
	// token fed here joins their window, as llama-server does with a prompt
	// before its first draw. What was drawn before is already in it — Pick put
	// it there — and is not fed again.
	s.sampler.Seed(ids[from:])
	turn := Turn{Prompt: len(ids) - from, Prefill: time.Since(start)}

	var answer strings.Builder
	start = time.Now()
	last := int32(-1)

	// A checkpoint that carries a prediction block drafts with it. Two tokens
	// then come out of one reading of the weights, which is what a token
	// actually costs; qwen35/speculate.go says how, and what a refused draft
	// has to undo.
	var draft speculator
	if sp, ok := s.model.(speculative); ok && sp.Speculate() && !s.noDraft {
		if d, err := sp.NewSpeculator(); err == nil {
			draft = d
		}
	}

	emit := func(id int32) error {
		piece := s.vocab.Piece(id, false)
		answer.WriteString(piece)
		if w == nil {
			return nil
		}
		_, err := io.WriteString(w, piece)
		return err
	}

	// pending is a token a speculative step already drew from the model's own
	// distribution. Drawing it again would be a second reading of the head,
	// which is the largest matrix in the model.
	pending := int32(-1)

	for turn.Generated < s.maxTokens && len(s.held) < s.maxContext {
		id := pending
		if id < 0 {
			s.model.Logits(hidden, s.logits)
			id = s.sampler.Pick(s.logits)
		}
		pending = -1
		turn.Generated++
		last = id
		if s.vocab.IsEOG(id) {
			break
		}
		if err := emit(id); err != nil {
			return turn, err
		}

		if draft != nil && turn.Generated+1 < s.maxTokens && len(s.held)+2 <= s.maxContext {
			next, h, err := draft.Step(id, hidden, len(s.held), s.sampler.Pick)
			if err != nil {
				return turn, err
			}
			s.held = append(s.held, id)
			// The last of what came back is the token after everything the
			// model has read, which is this loop's next `id`. Everything
			// before it is decided and goes into the cache.
			stop := false
			for _, tok := range next[:len(next)-1] {
				turn.Generated++
				last = tok
				if s.vocab.IsEOG(tok) {
					stop = true
					break
				}
				if err := emit(tok); err != nil {
					return turn, err
				}
				s.held = append(s.held, tok)
			}
			if stop {
				break
			}
			hidden = h
			pending = next[len(next)-1]
			continue
		}

		hidden = s.model.ForwardBatch([]int32{id}, len(s.held))[0]
		s.held = append(s.held, id)
	}
	if draft != nil {
		turn.Accepted, turn.Drafted = draft.Rate()
	}
	if draft != nil {
		turn.Accepted, turn.Drafted = draft.Rate()
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
