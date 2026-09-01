package qwen35

import (
	"fmt"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/vk"
)

// Speculative decoding with the checkpoint's own prediction block.
//
// The bargain is arithmetic. A token costs one reading of every weight in the
// model, and a pass carrying two tokens costs 1.056 of one carrying a single
// token — the weights are read once either way. So if the prediction block can
// guess the token after next often enough, two tokens can be had for the price
// of one pass plus the block, and the block is a sixty-fifth of the model.
//
// Measured on Qwen3.8-27B: the draft is the token the model itself chooses
// 81% of the time.
//
// The 1.056 is a Q4_0 kernel's, and a .golem checkpoint's is not. A trellis
// weight has to be decoded before it can be spent, so a second column costs
// what an activation load and a multiply-add cost on top of a decode that the
// two columns share — cmd/golemtune on Qwen3.8-27B's feed forward reads 89.8
// microseconds at one column and 112.6 at two, which is **1.25**, not 1.056.
// The bargain survives that on paper: at 74% acceptance a step returns 1.74
// tokens for 1.25 passes plus the block plus two extra readings of the logit
// head, which qwen35/cost_test.go measures at 2.2 microseconds against a
// token's 32.
//
// What it does not survive is being measured. On this card, with T3G weights,
// the drafting path is **bimodal** and the plain path is not: the same 128
// tokens from the same prompt take 4.6 seconds or 3.6 (and once 2.7), run to
// run, at the same 54-of-73 acceptance, while a token at a time holds 4.1
// within two per cent. In the slow mode drafting loses about a tenth; in the
// fast mode it wins about an eighth. The cause is not established — the card's
// clocks cannot be pinned without root here, and every attempt to warm it into
// one mode or the other landed in both.
//
// So: whether drafting pays is not a property of this code, and this comment
// is not going to claim it is. cmd/golem-cli has -draft, and -stats reports the
// acceptance rate; measure it on the card and the checkpoint in hand, several
// runs alternating, and believe the run-to-run spread before the mean. What is
// established, four runs of each with a warm-up turn in the same process:
//
//	                    a token at a time   drafting
//	  before the        19.25 / 19.29       26.30 / 26.35 / 26.41
//	  matvec was reworked
//	  after             30.96 / 31.20 /     27.85 / 28.06 / 35.39
//	                    31.55
//
// The plain path is where the kernel's 1.57 times went. The drafting path took
// only a little of it, because the pass it rides on is a pass of two and a pass
// of two gained less. That much is solid; which side of even it leaves the
// draft on, on this card, is not.
// What a recurrent model adds is the rollback. A refused draft leaves a key in
// the attention cache at a position it does not occupy, which the token that
// does occupy it overwrites — but it also leaves its contribution inside every
// delta net's state matrix, and there is nothing to overwrite that with. So the
// state is copied aside before the pass and copied back when the draft is
// refused, which vk/qwen_pipeline.go does on the card.

func argmax(v []float32) int32 {
	best, at := float32(-1e30), 0
	for i, x := range v {
		if x > best {
			best, at = x, i
		}
	}
	return int32(at)
}

// Speculator holds the buffers one speculative step needs.
type Speculator struct {
	m *Model

	eh     []float32
	draft  []float32
	verify [2][]float32
	embed  [2][]float32
	states [][]float32

	// Accepted and Drafted count what the run has done, so a caller can report
	// the acceptance rate the speedup actually came from.
	Accepted, Drafted int
}

// Rate is what the run has drafted and what it kept.
func (s *Speculator) Rate() (accepted, drafted int) { return s.Accepted, s.Drafted }

// Speculate reports whether the model can draft: it needs the prediction block
// and a card, because a draft made on the processor costs half a token and
// cannot save one.
func (m *Model) Speculate() bool {
	return m.HasMTP() && m.gpuPipe != nil && m.gpuPipe.HasMTP()
}

// NewSpeculator prepares the model to draft.
func (m *Model) NewSpeculator() (*Speculator, error) {
	if !m.Speculate() {
		return nil, fmt.Errorf("qwen35: this model cannot draft (prediction block on the card: %v)", m.HasMTP())
	}
	s := &Speculator{m: m}
	s.eh = make([]float32, m.Cfg.Dim*2)
	s.draft = make([]float32, m.Cfg.Vocab)
	for i := range s.verify {
		s.verify[i] = make([]float32, m.Cfg.Vocab)
		s.embed[i] = make([]float32, m.Cfg.Dim)
	}
	return s, nil
}

// Step advances the conversation by one or two tokens.
//
// token sits at pos and has already been decided; hidden is the trunk's
// output-normed state for the token before it. pick turns a distribution into a
// token and is the caller's own sampler.
//
// Every token returned is drawn from the model's own distribution: the draft
// decides only whether a second token comes back for free, never what either of
// them is. So this needs none of the accept-reject correction that speculating
// with a *different* model does — a refused draft costs the pass it rode on and
// nothing else.
//
// The draft itself is taken at the peak rather than through pick, because a
// sampler with a temperature drafting against itself would agree with itself
// about as often as chance allows.
//
// It returns the tokens decided after `token`, and the hidden state belonging
// to the last of them.
func (s *Speculator) Step(token int32, hidden []float32, pos int, pick func([]float32) int32) ([]int32, []float32, error) {
	return s.StepAt(token, hidden, Place{Pos: pos, T: pos, H: pos, W: pos}, pick)
}

// StepAt is Step for a conversation whose axes do not follow the cache index.
// The drafted column sits one further on every axis, which is what a text token
// after an image does: the picture is behind it, and what follows a picture
// advances the way text always has.
func (s *Speculator) StepAt(token int32, hidden []float32, at Place, pick func([]float32) int32) ([]int32, []float32, error) {
	m := s.m
	dim := m.Cfg.Dim

	// 1. Draft the token after `token`, from `token` and the state before it.
	m.W.TokenEmbd.Row(int(token), s.eh[:dim])
	nn.RMSNormPlain(s.eh[:dim], m.W.MTP.ENorm, m.Cfg.Eps)
	copy(s.eh[dim:], hidden)
	nn.RMSNormPlain(s.eh[dim:], m.W.MTP.HNorm, m.Cfg.Eps)

	h, err := m.gpuPipe.DraftMTPAt(s.eh, at.gpu())
	if err != nil {
		return nil, nil, err
	}
	m.Logits(h, s.draft)
	guess := argmax(s.draft)

	// 2. Run `token` and the guess through the trunk together. One reading of
	//    the weights answers both, which is the whole of the bargain.
	m.W.TokenEmbd.Row(int(token), s.embed[0])
	m.W.TokenEmbd.Row(int(guess), s.embed[1])
	next := at.Next()
	out, err := m.gpuPipe.ForwardSpeculativeAt(
		[][]float32{s.embed[0], s.embed[1]}, []vk.QwenPlace{at.gpu(), next.gpu()})
	if err != nil {
		return nil, nil, err
	}
	first := append([]float32(nil), out[0]...)
	second := append([]float32(nil), out[1]...)

	m.Logits(first, s.verify[0])
	truth := pick(s.verify[0])
	s.Drafted++

	if truth != guess {
		// The guess was wrong, so the second column never happened: the keys
		// it wrote will be overwritten, and the delta nets go back.
		if err := m.gpuPipe.RestoreState(); err != nil {
			return nil, nil, err
		}
		return []int32{truth}, first, nil
	}

	s.Accepted++
	m.Logits(second, s.verify[1])
	return []int32{truth, pick(s.verify[1])}, second, nil
}
