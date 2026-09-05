package sample

// The penalties are llama.cpp's, in its arithmetic and in its place in the
// chain: they come before top-k, they read a window of the identifiers already
// in the conversation, and they lower a logit that has been seen.
//
// What they must not do here is walk the row. llama.cpp penalises the whole
// candidate array because the array is already built; golem's is the
// vocabulary — 262144 entries — and building it costs more than the pass that
// produced the logits. Sampler.draw takes the shortcut instead, and window is
// what makes it exact.

// window is the identifiers the penalties are computed over — the last
// PenaltyLastN tokens of the conversation, prompt included — and the weights
// they are computed with.
//
// A PenaltyLastN of zero is the feature turned off. A negative one is the whole
// conversation: the ring is dropped and nothing is ever forgotten.
type window struct {
	last    int
	repeat  float32
	freq    float32
	present float32

	ring  []int32 // nil when the window is unbounded
	at    int
	full  bool
	count map[int32]int
}

func newWindow(p Params) window {
	w := window{last: p.PenaltyLastN, repeat: p.PenaltyRepeat,
		freq: p.PenaltyFreq, present: p.PenaltyPresent}
	if w.repeat == 0 {
		// A zero in the field means the caller never set it, and a repeat
		// penalty of zero would flatten every seen token to nothing.
		w.repeat = 1
	}
	if w.last > 0 {
		w.ring = make([]int32, w.last)
	}
	return w
}

// active reports whether anything here changes a logit. llama.cpp's own test,
// llama-sampler.cpp:2750.
func (w *window) active() bool {
	return w.last != 0 && (w.repeat != 1 || w.freq != 0 || w.present != 0)
}

// raises reports whether these weights can lift a logit rather than lower it.
// They can, and a caller is allowed to ask for it, but the shortcut in
// Sampler.draw rests on the opposite and has to stand aside when they do.
func (w *window) raises() bool {
	return w.repeat < 1 || w.freq < 0 || w.present < 0
}

// distinct is how many identifiers the window holds, which is how much wider
// than k the candidate list has to be for the shortcut to be exact.
func (w *window) distinct() int { return len(w.count) }

func (w *window) push(id int32) {
	if !w.active() {
		return
	}
	if w.count == nil {
		w.count = make(map[int32]int, w.last)
	}
	w.count[id]++
	if w.ring == nil {
		return
	}
	if w.full {
		old := w.ring[w.at]
		w.count[old]--
		if w.count[old] == 0 {
			delete(w.count, old)
		}
	}
	w.ring[w.at] = id
	w.at++
	if w.at == len(w.ring) {
		w.at, w.full = 0, true
	}
}

// apply is the arithmetic itself, llama-sampler.cpp:2686-2696. The
// multiplication below zero is not what the paper described: dividing a
// negative logit would make an unlikely token likelier, and this is the fix
// llama.cpp settled on.
func (w *window) apply(id int32, logit float32) float32 {
	n := w.count[id]
	if n == 0 {
		return logit
	}
	if logit <= 0 {
		logit *= w.repeat
	} else {
		logit /= w.repeat
	}
	return logit - float32(n)*w.freq - w.present
}

// Accept puts a token into the window. Pick calls it with whatever it draws,
// so a caller that hands Pick to a speculative step has nothing else to do.
func (s *Sampler) Accept(id int32) {
	s.pen.push(id)
	if s.con != nil {
		s.con.Accept(id)
	}
}

// Seed puts a whole run of tokens into the window at once, which is what the
// prompt is: llama-server feeds every prompt token to the sampler before the
// first draw (tools/server/server-context.cpp:254-260), so a repeat penalty
// sees the conversation and not only the answer.
func (s *Sampler) Seed(ids []int32) {
	for _, id := range ids {
		s.pen.push(id)
	}
}

// penalised is the slow road: a copy of the row with the penalties in it. Only
// weights that can raise a logit come here, and only the identifiers in the
// window are touched — the copy is the cost, not the arithmetic.
func (s *Sampler) penalised(logits []float32) []float32 {
	if !s.pen.active() {
		return logits
	}
	if cap(s.scratch) < len(logits) {
		s.scratch = make([]float32, len(logits))
	}
	row := s.scratch[:len(logits)]
	copy(row, logits)
	for id := range s.pen.count {
		if int(id) < len(row) {
			row[id] = s.pen.apply(id, row[id])
		}
	}
	return row
}

// penalise applies the window to candidates already selected, re-sorts them and
// cuts the list back to k.
//
// This is exact. A penalty can only lower a logit, so a token outside the k+u
// best of the raw row — u being the identifiers in the window — has at least k
// tokens above it that the window does not touch, and cannot climb into the
// penalised top k.
func (s *Sampler) penalise(kept []candidate, k int) []candidate {
	for i := range kept {
		kept[i].logit = s.pen.apply(kept[i].id, kept[i].logit)
	}
	sortCandidates(kept)
	if k > 0 && len(kept) > k {
		kept = kept[:k]
	}
	return kept
}

// widen is how many raw candidates the shortcut needs to keep k of them.
func widen(k, distinct, n int) int {
	if k <= 0 || k+distinct >= n {
		return 0 // the whole row; topK reads a zero that way
	}
	return k + distinct
}
