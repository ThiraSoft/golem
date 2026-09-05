package grammar

// The automaton: a set of pushdown stacks over the rules, advanced one code
// point at a time. Ported from llama-grammar.cpp — llama_grammar_advance_stack
// (853), llama_grammar_match_char (757), llama_grammar_match_partial_char (786)
// and llama_grammar_accept_chr (1016).
//
// A grammar is non-deterministic: alternatives and recursion mean several
// stacks are live at once, and a token is allowed if any of them survives it.

// position is where inside a rule a stack is. llama.cpp keeps a pointer here
// and pays for it when it clones a grammar; two integers cost nothing to copy,
// and Allows copies the whole state for every candidate it tests.
type position struct {
	rule uint32
	at   uint32
}

type stack []position

// Grammar is one grammar in the middle of one answer.
type Grammar struct {
	r    *Rules
	toks *Tokens

	stacks  []stack
	partial partialUTF8

	// first is the lead bytes any allowed token may start with, rebuilt when
	// the state moves and only if somebody asks. It is a superset: a byte may
	// be allowed here and the token still refused, never the other way round.
	first   [256]bool
	firstOK bool

	// Scratch, so that testing a candidate does not allocate a set of stacks
	// per code point. Two of them, because a step reads one and writes the
	// other.
	tryA, tryB []stack
	todo       []stack
	seen       map[uint64]bool
}

// New starts a grammar at the root of its rules.
func New(r *Rules, t *Tokens) *Grammar {
	g := &Grammar{r: r, toks: t, seen: map[uint64]bool{}}
	g.stacks = g.start(r.Root(), nil)
	return g
}

// start builds the stacks of a rule's alternatives.
func (g *Grammar) start(rule uint32, out []stack) []stack {
	elems := g.r.Rule(rule)
	for at := 0; at < len(elems); {
		var s stack
		if !endOfSequence(elems[at]) {
			s = stack{{rule: rule, at: uint32(at)}}
		}
		out = g.advance(s, out)
		for at < len(elems) && !endOfSequence(elems[at]) {
			at++
		}
		if at < len(elems) && elems[at].Kind == Alt {
			at++
			continue
		}
		break
	}
	return out
}

func endOfSequence(e Element) bool { return e.Kind == End || e.Kind == Alt }

func (g *Grammar) at(p position) Element { return g.r.Rule(p.rule)[p.at] }

// advance resolves rule references until every stack ends on a terminal, and
// appends the result to out. Without the deduplication a recursive rule
// produces the same stack over and over and the set grows without bound.
func (g *Grammar) advance(s stack, out []stack) []stack {
	todo := append(g.todo[:0], s)
	clear(g.seen)

	for len(todo) > 0 {
		cur := todo[len(todo)-1]
		todo = todo[:len(todo)-1]

		key := stackKey(cur)
		if g.seen[key] {
			continue
		}
		g.seen[key] = true

		if len(cur) == 0 {
			// An empty stack is the grammar satisfied: it is what lets an
			// end-of-turn token through, and it is kept once.
			out = appendUnique(out, cur)
			continue
		}

		top := cur[len(cur)-1]
		e := g.at(top)
		if e.Kind != RuleRef {
			out = appendUnique(out, cur)
			continue
		}

		// Every alternative of the rule referred to is a branch of its own.
		sub := g.r.Rule(uint32(e.Value))
		next := position{rule: top.rule, at: top.at + 1}
		hasNext := !endOfSequence(g.at(next))
		for at := 0; at < len(sub); {
			branch := make(stack, 0, len(cur)+1)
			branch = append(branch, cur[:len(cur)-1]...)
			if hasNext {
				branch = append(branch, next)
			}
			if !endOfSequence(sub[at]) {
				branch = append(branch, position{rule: uint32(e.Value), at: uint32(at)})
			}
			todo = append(todo, branch)
			for at < len(sub) && !endOfSequence(sub[at]) {
				at++
			}
			if at < len(sub) && sub[at].Kind == Alt {
				at++
				continue
			}
			break
		}
	}
	g.todo = todo[:0]
	return out
}

func appendUnique(out []stack, s stack) []stack {
	for _, have := range out {
		if sameStack(have, s) {
			return out
		}
	}
	return append(out, s)
}

func sameStack(a, b stack) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// stackKey is a stack as one number, for the deduplication above. A grammar
// with more than four thousand million elements in a rule is not a grammar.
func stackKey(s stack) uint64 {
	var h uint64 = 1469598103934665603
	for _, p := range s {
		h = (h ^ uint64(p.rule)) * 1099511628211
		h = (h ^ uint64(p.at)) * 1099511628211
	}
	return h
}

// step is the stacks that survive one code point.
func (g *Grammar) step(from []stack, r rune, out []stack) []stack {
	for _, s := range from {
		if len(s) == 0 {
			continue
		}
		top := s[len(s)-1]
		elems := g.r.Rule(top.rule)
		ok, next := matchChar(elems, int(top.at), r)
		if !ok {
			continue
		}
		grown := make(stack, 0, len(s))
		grown = append(grown, s[:len(s)-1]...)
		if !endOfSequence(elems[next]) {
			grown = append(grown, position{rule: top.rule, at: uint32(next)})
		}
		out = g.advance(grown, out)
	}
	return out
}

// matchChar answers whether the character class starting at `at` accepts r, and
// says where the class ends. The class is one positive or negative element
// followed by its ranges and alternates.
func matchChar(elems []Element, at int, r rune) (bool, int) {
	positive := elems[at].Kind == Char || elems[at].Kind == CharAny
	found := false
	for {
		switch {
		case at+1 < len(elems) && elems[at+1].Kind == CharRngUpper:
			found = found || (elems[at].Value <= r && r <= elems[at+1].Value)
			at += 2
		case elems[at].Kind == CharAny:
			found = true
			at++
		default:
			found = found || elems[at].Value == r
			at++
		}
		if at >= len(elems) || elems[at].Kind != CharAlt {
			break
		}
	}
	return found == positive, at
}

// matchPartial answers whether any completion of a character cut in two could
// satisfy the class at `at`. A token that ends mid-character is common in a
// byte-level BPE, and refusing it would refuse every accent and every ideogram.
func matchPartial(elems []Element, at int, p partialUTF8) bool {
	positive := elems[at].Kind == Char || elems[at].Kind == CharAny
	if p.remain < 0 || (p.remain == 1 && p.value < 2) {
		return false // a poisoned sequence, or a seven-bit character written long
	}

	low := p.value << (p.remain * 6)
	high := low | (1<<(p.remain*6) - 1)
	if low == 0 {
		switch p.remain {
		case 2:
			low = 1 << 11
		case 3:
			low = 1 << 16
		}
	}

	for {
		switch {
		case at+1 < len(elems) && elems[at+1].Kind == CharRngUpper:
			if uint32(elems[at].Value) <= high && low <= uint32(elems[at+1].Value) {
				return positive
			}
			at += 2
		case elems[at].Kind == CharAny:
			return true
		default:
			if low <= uint32(elems[at].Value) && uint32(elems[at].Value) <= high {
				return positive
			}
			at++
		}
		if at >= len(elems) || elems[at].Kind != CharAlt {
			break
		}
	}
	return !positive
}

// Done reports whether the grammar is satisfied — some stack has nothing left
// to match. While it is not, an end-of-turn token is refused: a draw at
// temperature would otherwise end the answer in the middle of an object and
// hand back a document with its braces open.
func (g *Grammar) Done() bool {
	if g.partial.remain != 0 {
		return false
	}
	for _, s := range g.stacks {
		if len(s) == 0 {
			return true
		}
	}
	return false
}

// Allows reports whether this token may come next. It changes nothing.
func (g *Grammar) Allows(id int32) bool {
	if int(id) >= len(g.toks.bytes) {
		return false
	}
	if g.toks.eog[id] {
		return g.Done()
	}
	piece := g.toks.bytes[id]
	if len(piece) == 0 {
		// A control token, or one that prints nothing. llama.cpp refuses these
		// outright (llama-grammar.cpp:1369): a model let loose on them loops on
		// tokens nobody can see.
		return false
	}
	runes, partial := decodeUTF8(piece, g.partial)
	if partial.remain < 0 {
		return false
	}

	live := append(g.tryA[:0], g.stacks...)
	other := g.tryB[:0]
	for _, r := range runes {
		next := g.step(live, r, other)
		if len(next) == 0 {
			g.tryA, g.tryB = live[:0], next[:0]
			return false
		}
		other = live[:0]
		live = next
	}
	g.tryA, g.tryB = live[:0], other[:0]

	if partial.remain > 0 {
		// The token ends mid-character: it is allowed only if some live stack
		// could accept a completion of it.
		for _, s := range live {
			if len(s) == 0 {
				continue
			}
			top := s[len(s)-1]
			if matchPartial(g.r.Rule(top.rule), int(top.at), partial) {
				return true
			}
		}
		return false
	}
	return true
}

// Accept advances the grammar over a token that was drawn. A token the grammar
// refuses leaves it where it was rather than in a state where nothing at all is
// allowed — the caller decides what to do about a draw it could not constrain.
func (g *Grammar) Accept(id int32) {
	if int(id) >= len(g.toks.bytes) || g.toks.eog[id] {
		return
	}
	piece := g.toks.bytes[id]
	if len(piece) == 0 {
		return
	}
	runes, partial := decodeUTF8(piece, g.partial)
	if partial.remain < 0 {
		return
	}

	live := append([]stack(nil), g.stacks...)
	for _, r := range runes {
		next := g.step(live, r, nil)
		if len(next) == 0 {
			return
		}
		live = next
	}
	g.stacks, g.partial, g.firstOK = live, partial, false
}

// FirstBytes is the lead bytes a token may start with here, which is what a
// full sweep of the vocabulary tests before it tests anything else. It is a
// superset — a byte may be allowed and the token still refused — and it is
// rebuilt only when somebody asks after the state has moved.
func (g *Grammar) FirstBytes() *[256]bool {
	if g.firstOK {
		return &g.first
	}
	g.first = [256]bool{}
	g.firstOK = true

	if g.partial.remain > 0 {
		// A character is open: only a continuation byte can come next.
		for b := 0x80; b < 0xC0; b++ {
			g.first[b] = true
		}
		return &g.first
	}

	for _, s := range g.stacks {
		if len(s) == 0 {
			continue
		}
		top := s[len(s)-1]
		elems := g.r.Rule(top.rule)
		at := int(top.at)
		if elems[at].Kind == CharNot || elems[at].Kind == CharAny {
			// A negated class refuses a few characters and allows the rest;
			// narrowing by first byte would buy nothing and could lie.
			for b := range g.first {
				g.first[b] = true
			}
			return &g.first
		}
		for {
			lo, hi := elems[at].Value, elems[at].Value
			if at+1 < len(elems) && elems[at+1].Kind == CharRngUpper {
				hi = elems[at+1].Value
				at += 2
			} else {
				at++
			}
			for b := int(leadByte(lo)); b <= int(leadByte(hi)); b++ {
				g.first[b] = true
			}
			if at >= len(elems) || elems[at].Kind != CharAlt {
				break
			}
		}
	}
	return &g.first
}

// leadByte is the first byte of a code point's UTF-8 encoding. Lead bytes grow
// with the code point, so the bytes of a range are the range of the bytes.
func leadByte(r rune) byte {
	switch {
	case r < 0x80:
		return byte(r)
	case r < 0x800:
		return byte(0xC0 | r>>6)
	case r < 0x10000:
		return byte(0xE0 | r>>12)
	default:
		return byte(0xF0 | r>>18)
	}
}

// FirstByte is the lead byte of a token's piece, for the sweep above.
func (g *Grammar) FirstByte(id int32) byte {
	if int(id) >= len(g.toks.first) {
		return 0
	}
	return g.toks.first[id]
}
