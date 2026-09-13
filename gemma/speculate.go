package gemma

// Speculative decoding with the assistant.
//
// The bargain is qwen35/speculate.go's: a pass carrying two tokens reads the
// weights once, as a pass carrying one does, so a right guess is a second token
// for the price of the assistant. The assistant here is a model of its own
// rather than a block of the checkpoint, and it reads the target's cache rather
// than keeping one, so a refused guess leaves nothing behind but a key and a
// value at a position the next token overwrites.

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
	return m.assistant.NewSpeculator(), nil
}

// Speculator holds the buffers one speculative step needs.
type Speculator struct {
	m      *Model
	a      *Assistant
	draft  []float32
	verify [2][]float32
	tokens [2]int32

	// Accepted and Drafted count what the run has done, so a caller can report
	// the acceptance rate the speedup actually came from.
	Accepted, Drafted int
}

// NewSpeculator prepares the assistant to draft for its model.
func (a *Assistant) NewSpeculator() *Speculator {
	s := &Speculator{m: a.target, a: a, draft: make([]float32, a.Cfg.Vocab)}
	for i := range s.verify {
		s.verify[i] = make([]float32, a.target.Cfg.Vocab)
	}
	return s
}

// Rate is what the run has drafted and what it kept.
func (s *Speculator) Rate() (accepted, drafted int) { return s.Accepted, s.Drafted }

// Step advances the conversation by one or two tokens, with qwen35's contract:
// token sits at pos and has already been decided, hidden is the model's normed
// state for the position before it, and pick is the caller's own sampler.
//
// Every token returned is drawn from the model's own distribution; the guess
// only decides whether a second one comes back from the same pass. The guess is
// the assistant's peak, which is what llama.cpp takes too (top-k 10, the first).
//
// It returns the tokens decided after `token`, and the state belonging to the
// last of them.
func (s *Speculator) Step(token int32, hidden []float32, pos int, pick func([]float32) int32) ([]int32, []float32, error) {
	s.a.Draft(token, hidden, pos, s.draft)
	guess := Argmax(s.draft)

	s.tokens = [2]int32{token, guess}
	out := s.m.ForwardBatch(s.tokens[:], pos)
	first := append([]float32(nil), out[0]...)
	second := append([]float32(nil), out[1]...)

	s.m.Logits(first, s.verify[0])
	truth := pick(s.verify[0])
	s.Drafted++
	if truth != guess {
		// The guess's key and value sit at pos+1, where the next token
		// writes its own.
		return []int32{truth}, first, nil
	}
	s.Accepted++
	s.m.Logits(second, s.verify[1])
	return []int32{truth, pick(s.verify[1])}, second, nil
}
