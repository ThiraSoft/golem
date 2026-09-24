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

// SetDraftDepth says how many tokens a step guesses, which has to be said
// before the upload; see Model.draftDepth.
func (m *Model) SetDraftDepth(n int) { m.draftDepth = n }

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

	// depth is how many tokens a step guesses.
	depth int
	// peak says the caller's pick is the row's peak and nothing else, so the
	// card can find it where the row is written; see SetPeak.
	peak bool

	eh     []float32
	draft  []float32
	verify [][]float32
	embed  [][]float32

	// Accepted and Drafted count what the run has done, so a caller can report
	// the acceptance rate the speedup actually came from.
	Accepted, Drafted int
}

// SetPeak says whether the caller's pick is plain greedy — no penalty, no
// grammar — for the steps that follow. When it is, the verifying pass's
// answers are found on the card and pick is not called.
func (s *Speculator) SetPeak(on bool) { s.peak = on }

// Rate is what the run has drafted and what it kept.
func (s *Speculator) Rate() (accepted, drafted int) { return s.Accepted, s.Drafted }

// Span is the most positions a step writes: the token and its guesses.
func (s *Speculator) Span() int { return s.depth + 1 }

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
	s := &Speculator{m: m, depth: max(m.draftDepth, 1)}
	s.eh = make([]float32, m.Cfg.Dim*2)
	s.draft = make([]float32, m.Cfg.Vocab)
	s.verify = make([][]float32, s.depth+1)
	s.embed = make([][]float32, s.depth+1)
	for i := range s.verify {
		s.verify[i] = make([]float32, m.Cfg.Vocab)
		s.embed[i] = make([]float32, m.Cfg.Dim)
	}
	return s, nil
}

// Step advances the conversation by one token or more.
//
// token sits at pos and has already been decided; hidden is the trunk's
// output-normed state for the token before it. pick turns a distribution into a
// token and is the caller's own sampler.
//
// Every token returned is drawn from the model's own distribution: the drafts
// decide only how many tokens come back for one pass, never what any of them
// is. So this needs none of the accept-reject correction that speculating with
// a *different* model does — a refused draft costs the pass it rode on and
// nothing else.
//
// The drafts themselves are taken at the peak rather than through pick,
// because a sampler with a temperature drafting against itself would agree
// with itself about as often as chance allows.
//
// It returns the tokens decided after `token`, and the hidden state belonging
// to the last of them.
func (s *Speculator) Step(token int32, hidden []float32, pos int, pick func([]float32) int32) ([]int32, []float32, error) {
	return s.StepAt(token, hidden, Place{Pos: pos, T: pos, H: pos, W: pos}, pick)
}

// StepAt is Step for a conversation whose axes do not follow the cache index.
// The drafted columns sit one further on every axis each, which is what text
// after an image does: the picture is behind it, and what follows a picture
// advances the way text always has.
func (s *Speculator) StepAt(token int32, hidden []float32, at Place, pick func([]float32) int32) ([]int32, []float32, error) {
	m := s.m
	dim := m.Cfg.Dim

	// 1. Draft depth tokens, the prediction block run on its own output: the
	//    first from `token` and the trunk's state before it, every later one
	//    from the guess before it and the block's own output-normed state,
	//    which is the same kind of vector the trunk hands it.
	places := make([]Place, s.depth+1)
	places[0] = at
	for i := 1; i <= s.depth; i++ {
		places[i] = places[i-1].Next()
	}
	guesses := make([]int32, s.depth)
	prev, h := token, hidden
	for i := 0; i < s.depth; i++ {
		m.W.TokenEmbd.Row(int(prev), s.eh[:dim])
		nn.RMSNormPlain(s.eh[:dim], m.W.MTP.ENorm, m.Cfg.Eps)
		copy(s.eh[dim:], h)
		nn.RMSNormPlain(s.eh[dim:], m.W.MTP.HNorm, m.Cfg.Eps)
		if m.gpuPipe.HasDraftHead() {
			tok, out, err := m.gpuPipe.DraftTokenAt(s.eh, places[i].gpu())
			if err != nil {
				return nil, nil, err
			}
			h = append(h[:0:0], out...)
			guesses[i] = tok
			prev = guesses[i]
			continue
		}
		out, err := m.gpuPipe.DraftMTPAt(s.eh, places[i].gpu())
		if err != nil {
			return nil, nil, err
		}
		h = append([]float32(nil), out...)
		m.Logits(h, s.draft)
		guesses[i] = argmax(s.draft)
		prev = guesses[i]
	}

	// 2. Run `token` and the guesses through the trunk together. One reading
	//    of the weights answers all of them, which is the whole of the bargain.
	m.W.TokenEmbd.Row(int(token), s.embed[0])
	for i, g := range guesses {
		m.W.TokenEmbd.Row(int(g), s.embed[i+1])
	}
	gpu := make([]vk.QwenPlace, len(places))
	for i, pl := range places {
		gpu[i] = pl.gpu()
	}
	out, err := m.gpuPipe.ForwardSpeculativeAt(s.embed, gpu)
	if err != nil {
		return nil, nil, err
	}
	states := make([][]float32, len(out))
	for i := range out {
		states[i] = append([]float32(nil), out[i]...)
	}
	var peaks []int32
	if s.peak && m.gpuPipe.HasDraftHead() {
		if peaks, err = m.gpuPipe.PeaksOf(len(out)); err != nil {
			return nil, nil, err
		}
	} else {
		m.LogitsBatch(states, s.verify)
	}

	// 3. Keep the guesses the model agrees with, up to the first it does not.
	decided := make([]int32, 0, s.depth+1)
	kept := 0
	for i := 0; ; i++ {
		var truth int32
		if peaks != nil {
			truth = peaks[i]
		} else {
			truth = pick(s.verify[i])
		}
		decided = append(decided, truth)
		if i == s.depth || truth != guesses[i] {
			break
		}
		kept++
	}
	s.Drafted += s.depth
	s.Accepted += kept
	if kept < s.depth {
		// The columns past the last kept guess never happened: the keys they
		// wrote will be overwritten, and the delta nets go back to the copy
		// taken at the start of the first of them.
		if err := m.gpuPipe.RestoreStateAt(kept); err != nil {
			return nil, nil, err
		}
	}
	return decided, states[kept], nil
}
