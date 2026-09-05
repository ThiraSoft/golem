package sample

import "testing"

// A constraint that allows a named set of identifiers, and counts what it was
// asked. The counts are the point: the chain is supposed to ask about the token
// it drew and nothing else, until that token is refused.
type set struct {
	ok       map[int32]bool
	asked    int
	accepted []int32
}

func allow(ids ...int32) *set {
	s := &set{ok: map[int32]bool{}}
	for _, id := range ids {
		s.ok[id] = true
	}
	return s
}

func (s *set) Allows(id int32) bool { s.asked++; return s.ok[id] }
func (s *set) Accept(id int32)      { s.accepted = append(s.accepted, id) }
func (s *set) FirstByte(int32) byte { return 0 }

func (s *set) FirstBytes() *[256]bool {
	var all [256]bool
	for i := range all {
		all[i] = true
	}
	return &all
}

func TestAnAllowedDrawIsNotSecondGuessed(t *testing.T) {
	c := allow(0, 1, 2)
	s := New(Params{Temperature: 0})
	s.Constrain(c)

	if got := s.Pick([]float32{3, 2, 1}); got != 0 {
		t.Fatalf("the draw named %d", got)
	}
	if c.asked != 1 {
		t.Fatalf("the constraint was asked %d times about a draw it allowed", c.asked)
	}
}

func TestARefusedDrawIsRedrawnAmongTheAllowed(t *testing.T) {
	s := New(Params{Temperature: 0})
	s.Constrain(allow(2))
	if got := s.Pick([]float32{3, 2, 1}); got != 2 {
		t.Fatalf("the draw named %d, and only 2 is allowed", got)
	}
}

// The prefix widens until it holds k allowed candidates. Here the allowed
// tokens sit at the bottom of a long row, so the first prefix holds none.
func TestTheChainWidensUntilItHasEnough(t *testing.T) {
	const n = 4096
	row := make([]float32, n)
	for i := range row {
		row[i] = float32(n - i)
	}
	s := New(Params{Temperature: 1, TopK: 4, TopP: 1, Seed: 3})
	s.Constrain(allow(4000, 4001, 4002, 4003))

	for i := 0; i < 16; i++ {
		got := s.Pick(row)
		if got < 4000 || got > 4003 {
			t.Fatalf("draw %d named %d, which the constraint refuses", i, got)
		}
	}
}

// A constraint that allows nothing must not hang the answer: the caller gets an
// unconstrained token and can decide to stop.
func TestAConstraintThatAllowsNothingStillNamesAToken(t *testing.T) {
	s := New(Params{Temperature: 1, TopK: 4, TopP: 1, Seed: 3})
	s.Constrain(allow())
	if got := s.Pick([]float32{3, 2, 1}); got < 0 || got > 2 {
		t.Fatalf("the draw named %d, which is not in the row", got)
	}
}

// Whatever is drawn is accepted into the constraint, once, in order.
func TestTheConstraintSeesEveryDrawnToken(t *testing.T) {
	c := allow(0, 1, 2)
	s := New(Params{Temperature: 1, TopK: 3, TopP: 1, Seed: 5})
	s.Constrain(c)
	for i := 0; i < 5; i++ {
		s.Pick([]float32{1, 1, 1})
	}
	if len(c.accepted) != 5 {
		t.Fatalf("five draws left %d accepted tokens: %v", len(c.accepted), c.accepted)
	}
}

// The penalties and the constraint have to hold at the same time: the redraw
// reads penalised logits, not raw ones.
func TestAConstrainedRedrawIsStillPenalised(t *testing.T) {
	s := New(Params{Temperature: 0, PenaltyLastN: 64, PenaltyRepeat: 1, PenaltyPresent: 100})
	s.Constrain(allow(1, 2))
	s.Seed([]int32{1})

	// Identifier 0 is highest and refused; 1 is next but the window has it,
	// which costs it a hundred; so 2 is what is left.
	if got := s.Pick([]float32{10, 9, 8}); got != 2 {
		t.Fatalf("the draw named %d, and the penalty should have pushed 1 below 2", got)
	}
}
