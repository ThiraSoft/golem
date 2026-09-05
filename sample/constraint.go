package sample

// A constraint is something that says which tokens may come next — a grammar,
// in practice, though nothing here knows that. It is llama.cpp's arrangement
// (common/sampling.cpp:576-618) and its laziness with it: the chain draws
// first, the drawn token alone is tested, and only a refusal costs anything.
//
// The expensive road is a sweep of the vocabulary, and it is walked as little
// as possible. The k best allowed candidates are the first k allowed
// candidates in descending order, so a prefix of the sorted row is enough: take
// the top k, filter, and while too few survive, widen the prefix. On an
// ordinary state it stops at the first one.

// Constraint is the part of a grammar the sampler drives.
type Constraint interface {
	// Allows reports whether this token may come next, changing nothing.
	Allows(id int32) bool
	// Accept advances over a token that was drawn.
	Accept(id int32)
	// FirstBytes is the lead bytes an allowed token may start with, as a
	// superset: a sweep tests one of these before it asks Allows anything.
	FirstBytes() *[256]bool
	// FirstByte is a token's own lead byte, or zero for a token whose piece
	// says nothing about whether it is allowed.
	FirstByte(id int32) byte
}

// Constrain hands the sampler a constraint to draw inside. Every token Pick
// names from here on is one the constraint allowed, and is accepted into it.
func (s *Sampler) Constrain(c Constraint) { s.con = c }

// allowed returns the k best candidates the constraint allows, penalised and
// sorted, or nothing at all when it allows none of them.
func (s *Sampler) allowed(logits []float32, k int, penalise bool) []candidate {
	target := k
	if penalise {
		target += s.pen.distinct()
	}
	first := s.con.FirstBytes()

	width := k
	if width <= 0 || width > len(logits) {
		width = len(logits)
	}
	for {
		kept := s.topK(logits, widthOrAll(width, len(logits)))
		s.keep = s.keep[:0]
		for _, c := range kept {
			// The byte first, because it is one comparison against a table
			// and the walk behind Allows is not. In a state that refuses
			// nearly everything — the start of an object, just after a comma —
			// this is what keeps a sweep of the vocabulary affordable.
			if first[s.con.FirstByte(c.id)] && s.con.Allows(c.id) {
				s.keep = append(s.keep, c)
			}
		}
		if len(s.keep) >= target || width >= len(logits) {
			break
		}
		width *= 4
	}
	if len(s.keep) == 0 {
		return nil
	}
	if penalise {
		return s.penalise(s.keep, k)
	}
	if k > 0 && len(s.keep) > k {
		s.keep = s.keep[:k]
	}
	return s.keep
}

// widthOrAll turns a prefix width into the k topK reads, where anything at or
// past the row is the whole row.
func widthOrAll(width, n int) int {
	if width >= n {
		return 0
	}
	return width
}
