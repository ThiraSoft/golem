// Package sample turns a row of logits into a token.
//
// The chain is the one llama.cpp runs by default, in its order: top-k cuts the
// row down to the most likely few, top-p cuts it again at a share of the mass,
// temperature reshapes what is left, and a draw from the resulting distribution
// names the token. A temperature of zero short-circuits all of it and takes the
// highest logit.
//
// Nothing here knows what model produced the row.
package sample

import (
	"math"
	"math/rand/v2"
	"sort"
)

type Params struct {
	// Temperature divides the logits before the draw. Zero or less is greedy.
	Temperature float32
	// TopK keeps that many candidates. Zero or less keeps all of them.
	TopK int
	// TopP keeps the shortest prefix of the sorted candidates whose
	// probabilities reach it. One or more keeps all of them.
	TopP float32
	// Seed fixes the run: the same seed over the same logits draws the same
	// tokens.
	Seed uint64

	// PenaltyLastN is how many of the conversation's last tokens the three
	// penalties below read. Zero turns them off; a negative one reads the
	// whole conversation.
	PenaltyLastN int
	// PenaltyRepeat divides the logit of a token in that window, or
	// multiplies it when the logit is negative. One is no penalty.
	PenaltyRepeat float32
	// PenaltyFreq is subtracted once per appearance in the window,
	// PenaltyPresent once for any appearance at all. Zero is no penalty.
	PenaltyFreq    float32
	PenaltyPresent float32
}

// Defaults are the values Gemma 4's own file declares under general.sampling,
// with llama.cpp's own defaults for the penalties it does not declare.
func Defaults() Params {
	return Params{Temperature: 1, TopK: 64, TopP: 0.95,
		PenaltyLastN: 64, PenaltyRepeat: 1}
}

// candidate is one token and its score, carried through the chain together
// because the sorting loses the position.
type candidate struct {
	id    int32
	logit float32
}

type Sampler struct {
	p    Params
	rng  *rand.Rand
	pool []candidate // reused across draws; a vocabulary row is large
	prob []float32
	// pen is what the penalties read: the identifiers already in the
	// conversation. A sampler with no penalties never allocates it.
	pen window
	// scratch holds a penalised copy of the row, for the weights the
	// shortcut cannot take. Nothing else allocates a row here.
	scratch []float32
}

func New(p Params) *Sampler {
	return &Sampler{
		p:   p,
		pen: newWindow(p),
		// PCG rather than the global source: a sampler is a value with a seed,
		// and two of them must not interfere.
		rng: rand.New(rand.NewPCG(p.Seed, p.Seed^0x9e3779b97f4a7c15)),
	}
}

// Greedy is the highest logit. Ties go to the lower identifier, as llama.cpp
// does.
func Greedy(logits []float32) int32 {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return int32(best)
}

// Softmax turns logits into probabilities, shifted by the largest so that a row
// of several hundred does not overflow the exponential.
func Softmax(logits []float32) []float32 {
	out := make([]float32, len(logits))
	softmaxInto(logits, out)
	return out
}

func softmaxInto(logits, out []float32) {
	if len(logits) == 0 {
		return
	}
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - max))
		out[i] = float32(e)
		sum += e
	}
	for i := range out {
		out[i] = float32(float64(out[i]) / sum)
	}
}

// Pick names a token. The logits are read, never written.
//
// Whatever it names joins the penalty window, so a caller that hands Pick to a
// speculative step has nothing else to keep up to date.
func (s *Sampler) Pick(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	id := s.draw(logits)
	s.Accept(id)
	return id
}

// draw is Pick without the bookkeeping.
func (s *Sampler) draw(logits []float32) int32 {
	switch {
	case !s.pen.active():
		return s.plainDraw(logits)
	case s.pen.raises():
		// Weights that can lift a logit rather than lower it. The shortcut
		// below cannot see those coming, so the row is penalised in full.
		return s.plainDraw(s.penalised(logits))
	}

	// The shortcut: k+u raw candidates, penalised and cut back to k, which is
	// the set a full pass would have left.
	k := s.p.TopK
	if s.p.Temperature <= 0 {
		k = 1
	}
	kept := s.penalise(s.topK(logits, widen(k, s.pen.distinct(), len(logits))), k)
	return s.pickFrom(kept)
}

// plainDraw is the chain over a row that needs no penalising.
func (s *Sampler) plainDraw(logits []float32) int32 {
	if s.p.Temperature <= 0 {
		return Greedy(logits)
	}
	return s.pickFrom(s.topK(logits, s.p.TopK))
}

// pickFrom is top-p, temperature and the draw, over candidates already chosen.
func (s *Sampler) pickFrom(kept []candidate) int32 {
	if len(kept) == 0 {
		return 0
	}
	if s.p.Temperature <= 0 {
		return kept[0].id
	}
	kept = s.topP(kept)
	if len(kept) == 1 {
		return kept[0].id
	}

	if cap(s.prob) < len(kept) {
		s.prob = make([]float32, len(kept))
	}
	prob := s.prob[:len(kept)]
	for i, c := range kept {
		prob[i] = c.logit / s.p.Temperature
	}
	softmaxInto(prob, prob)

	// The draw. The last candidate catches whatever the accumulated rounding
	// leaves short of one.
	r := float32(s.rng.Float64())
	var cum float32
	for i, p := range prob {
		cum += p
		if r < cum {
			return kept[i].id
		}
	}
	return kept[len(kept)-1].id
}

// topK returns the k candidates worth keeping, highest logit first, ties to the
// lower identifier. A k of zero or less keeps the row. When k is smaller than it — which it is, for a
// vocabulary of a quarter of a million — the selection runs through a heap of
// k entries rather than sorting everything.
func (s *Sampler) topK(logits []float32, k int) []candidate {
	if k <= 0 || k > len(logits) {
		k = len(logits)
	}
	if len(logits) <= k {
		if cap(s.pool) < len(logits) {
			s.pool = make([]candidate, len(logits))
		}
		res := s.pool[:len(logits)]
		for i, v := range logits {
			res[i] = candidate{id: int32(i), logit: v}
		}
		sortCandidates(res)
		return res
	}
	if cap(s.pool) < k {
		s.pool = make([]candidate, k)
	}
	heap := s.pool[:k]
	for i := 0; i < k; i++ {
		heap[i] = candidate{id: int32(i), logit: logits[i]}
	}
	buildMinHeap(heap)
	minLogit := heap[0].logit

	for i := k; i < len(logits); i++ {
		v := logits[i]
		if v < minLogit {
			continue
		}
		c := candidate{id: int32(i), logit: v}
		if better(heap[0], c) {
			continue
		}
		heap[0] = c
		siftDown(heap, 0)
		minLogit = heap[0].logit
	}
	sortCandidates(heap)
	return heap
}

func sortCandidates(c []candidate) {
	sort.Slice(c, func(i, j int) bool { return better(c[i], c[j]) })
}

// better is the order the whole package uses: a higher logit wins, and an equal
// one goes to the lower identifier.
func better(a, b candidate) bool {
	if a.logit != b.logit {
		return a.logit > b.logit
	}
	return a.id < b.id
}

func buildMinHeap(h []candidate) {
	for i := len(h)/2 - 1; i >= 0; i-- {
		siftDown(h, i)
	}
}

// siftDown keeps the worst candidate at the root.
func siftDown(h []candidate, i int) {
	for {
		worst := i
		for _, child := range [2]int{2*i + 1, 2*i + 2} {
			if child < len(h) && better(h[worst], h[child]) {
				worst = child
			}
		}
		if worst == i {
			return
		}
		h[i], h[worst] = h[worst], h[i]
		i = worst
	}
}

// topP cuts the sorted candidates where their probabilities reach p, keeping
// the one that crosses it and never fewer than one.
func (s *Sampler) topP(kept []candidate) []candidate {
	if s.p.TopP >= 1 || len(kept) < 2 {
		return kept
	}
	if cap(s.prob) < len(kept) {
		s.prob = make([]float32, len(kept))
	}
	prob := s.prob[:len(kept)]
	for i, c := range kept {
		prob[i] = c.logit
	}
	softmaxInto(prob, prob)

	var cum float32
	for i, p := range prob {
		cum += p
		if cum >= s.p.TopP {
			return kept[:i+1]
		}
	}
	return kept
}
