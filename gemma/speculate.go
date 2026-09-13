package gemma

// Speculative decoding with the assistant.
//
// The bargain is qwen35/speculate.go's: a pass carrying two tokens reads the
// weights once, as a pass carrying one does, so a right guess is a second token
// for the price of the assistant. The assistant here is a model of its own
// rather than a block of the checkpoint, and it reads the target's cache rather
// than keeping one, so a refused guess leaves nothing behind but keys and
// values at positions the next tokens overwrite.
//
// It guesses up to Depth tokens a step, as llama.cpp's --spec-draft-n-max does:
// the second is drawn from the assistant's own state after the first, and the
// target checks all of them in one pass.

import "fmt"

// OpenAssistant gives the model the drafter published beside it. The model
// closes it.
func (m *Model) OpenAssistant(path string) error {
	a, err := OpenAssistant(path, m)
	if err != nil {
		return err
	}
	// A model already on a card takes the assistant there now; one that is
	// not yet takes it when UseVulkanStack runs.
	if m.stack != nil {
		if err := a.useVulkan(); err != nil {
			a.Close()
			return err
		}
	}
	if m.assistant != nil {
		m.assistant.Close()
	}
	m.assistant = a
	return nil
}

// Speculate reports whether the model can draft: it needs an assistant, and
// the assistant where the model's caches are — both on the processor, or both
// on the card.
func (m *Model) Speculate() bool {
	return m.assistant != nil && (m.stack == nil) == (m.assistant.stack == nil)
}

// NewSpeculator prepares the model to draft with its assistant.
func (m *Model) NewSpeculator() (*Speculator, error) {
	if !m.Speculate() {
		return nil, fmt.Errorf("gemma: this model cannot draft (assistant: %v, blocks on a card: %v)",
			m.assistant != nil, m.stack != nil)
	}
	depth := m.draftDepth
	if depth == 0 {
		depth = DefaultDepth
	}
	return m.assistant.NewSpeculator(depth), nil
}

// SetDraftDepth says how many tokens the speculators made from now on guess a
// step; zero is DefaultDepth.
func (m *Model) SetDraftDepth(n int) { m.draftDepth = n }

// DefaultDepth is how many tokens a step guesses unless told otherwise.
//
// One, measured. 12B QAT on the card, greedy, two runs each: a short answer
// 91.9 / 74.0 t/s at one, 79.7 / 78.8 at two, 81.2 / 82.8 at three; a long
// one 86.8 / 82.8, 80.1 / 88.0, 85.3 / 78.9. The second guess is kept less
// often than the first (65% of drafts against 81% on the short answer) and
// costs a whole draft — the projection, four blocks, and a head of 262144
// rows scored and scanned — so it pays for itself and no more.
const DefaultDepth = 1

// MaxDepth is the most a step guesses. The verifying pass is Depth+1 columns,
// and the card's narrow binaries stop at four.
const MaxDepth = 3

// Speculator holds the buffers one speculative step needs.
type Speculator struct {
	m      *Model
	a      *Assistant
	depth  int
	draft  []float32
	verify [][]float32
	tokens []int32

	// Accepted and Drafted count what the run has done, so a caller can report
	// the acceptance rate the speedup actually came from.
	Accepted, Drafted int
}

// NewSpeculator prepares the assistant to draft for its model, depth tokens a
// step.
func (a *Assistant) NewSpeculator(depth int) *Speculator {
	depth = min(max(depth, 1), MaxDepth)
	s := &Speculator{m: a.target, a: a, depth: depth, draft: make([]float32, a.Cfg.Vocab)}
	s.verify = rows(depth+1, a.target.Cfg.Vocab)
	s.tokens = make([]int32, 0, depth+1)
	return s
}

// SetDepth changes how many tokens a step guesses, within 1 and MaxDepth.
func (s *Speculator) SetDepth(depth int) {
	depth = min(max(depth, 1), MaxDepth)
	if depth+1 > len(s.verify) {
		s.verify = append(s.verify, rows(depth+1-len(s.verify), s.m.Cfg.Vocab)...)
	}
	s.depth = depth
}

// Span is the most tokens a step adds after the one it is given, which is
// what a caller has to leave room for in the context.
func (s *Speculator) Span() int { return s.depth + 1 }

// Rate is what the run has drafted and what it kept.
func (s *Speculator) Rate() (accepted, drafted int) { return s.Accepted, s.Drafted }

// Step advances the conversation by one to Depth+1 tokens, with qwen35's
// contract: token sits at pos and has already been decided, hidden is the
// model's normed state for the position before it, and pick is the caller's
// own sampler.
//
// Every token returned is drawn from the model's own distribution; the guesses
// only decide how many come back from the same pass. A guess is the
// assistant's peak, which is what llama.cpp takes too (top-k 10, the first).
//
// It returns the tokens decided after `token`, and the state belonging to the
// last of them.
func (s *Speculator) Step(token int32, hidden []float32, pos int, pick func([]float32) int32) ([]int32, []float32, error) {
	s.tokens = append(s.tokens[:0], token)
	s.a.Draft(token, hidden, pos, s.draft)
	s.tokens = append(s.tokens, Argmax(s.draft))
	for len(s.tokens) <= s.depth {
		s.a.DraftNext(s.tokens[len(s.tokens)-1], pos, s.draft)
		s.tokens = append(s.tokens, Argmax(s.draft))
	}

	out := s.m.ForwardBatch(s.tokens, pos)
	var decided []int32
	for i := range s.tokens {
		s.m.Logits(out[i], s.verify[i])
		truth := pick(s.verify[i])
		decided = append(decided, truth)
		if i+1 == len(s.tokens) {
			// Every guess was right, and this is the token after the last
			// of them: nothing to check it against.
			break
		}
		s.Drafted++
		if truth != s.tokens[i+1] {
			// A refused guess and everything after it never happened: their
			// keys and values sit at positions the next tokens overwrite.
			break
		}
		s.Accepted++
	}
	return decided, append([]float32(nil), out[len(decided)-1]...), nil
}
