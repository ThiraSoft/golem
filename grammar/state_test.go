package grammar

import (
	"strings"
	"testing"
)

// A vocabulary of whole pieces: each string is one token. That is enough to
// script what a tokenizer would hand a grammar, splits of a character across
// two tokens included.
func tokensOf(pieces ...string) *Tokens {
	all := append([]string{""}, pieces...) // identifier 0 prints nothing
	all = append(all, "<eos>")
	eog := len(all) - 1
	return NewTokens(len(all), func(id int32) string { return all[id] },
		func(id int32) bool { return int(id) == eog })
}

func idOf(t *Tokens, piece string) int32 {
	for id, b := range t.bytes {
		if string(b) == piece {
			return int32(id)
		}
	}
	panic("no token " + piece)
}

func feed(t *testing.T, g *Grammar, toks *Tokens, pieces ...string) {
	t.Helper()
	for _, piece := range pieces {
		id := idOf(toks, piece)
		if !g.Allows(id) {
			t.Fatalf("the grammar refused %q", piece)
		}
		g.Accept(id)
	}
}

func mustParse(t *testing.T, src string) *Rules {
	t.Helper()
	r, err := Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

const objectGrammar = `
root   ::= "{" ws "\"n\"" ws ":" ws number ws "}"
number ::= [0-9]+
ws     ::= [ \t\n]*
`

func TestAValidDocumentGoesThroughTokenByToken(t *testing.T) {
	toks := tokensOf("{", "\"n\"", ":", " ", "12", "}", "x")
	g := New(mustParse(t, objectGrammar), toks)
	feed(t, g, toks, "{", "\"n\"", ":", " ", "12", "}")
	if !g.Done() {
		t.Fatal("the document closed and the grammar does not think it is done")
	}
}

func TestTheTokenThatBreaksItIsRefused(t *testing.T) {
	toks := tokensOf("{", "\"n\"", ":", " ", "12", "}", "x")
	g := New(mustParse(t, objectGrammar), toks)
	feed(t, g, toks, "{")
	if g.Allows(idOf(toks, "x")) {
		t.Fatal("a letter was allowed where the grammar wants a quoted name")
	}
	if g.Allows(idOf(toks, "}")) {
		t.Fatal("the object closed before it held anything")
	}
}

// A byte-level tokenizer splits a character wherever it likes. A grammar that
// refused the first half of an accent would refuse every accented word.
func TestACharacterSplitAcrossTwoTokensIsAccepted(t *testing.T) {
	r := mustParse(t, `root ::= [é]`)
	toks := tokensOf("\xc3", "\xa9", "\xc3\xa9", "z")
	g := New(r, toks)

	if !g.Allows(idOf(toks, "\xc3")) {
		t.Fatal("the first byte of é was refused")
	}
	g.Accept(idOf(toks, "\xc3"))
	if g.Done() {
		t.Fatal("half a character satisfied the grammar")
	}
	if !g.Allows(idOf(toks, "\xa9")) {
		t.Fatal("the second byte of é was refused")
	}
	g.Accept(idOf(toks, "\xa9"))
	if !g.Done() {
		t.Fatal("the character completed and the grammar is not done")
	}
}

func TestAHalfCharacterThatCannotBeCompletedIsRefused(t *testing.T) {
	// [à] is 0xC3 0xA0; the piece 0xC3 could still become it, 0xC5 could not.
	toks := tokensOf("\xc3", "\xc5")
	g := New(mustParse(t, `root ::= [à]`), toks)
	if !g.Allows(idOf(toks, "\xc3")) {
		t.Fatal("the lead byte of à was refused")
	}
	if g.Allows(idOf(toks, "\xc5")) {
		t.Fatal("a lead byte that cannot become à was allowed")
	}
}

// The failure that costs the most and shows the least: a draw at temperature
// takes the end-of-turn token in the middle of an object, and the answer comes
// back with its braces open.
func TestTheEndOfTurnIsRefusedUntilTheGrammarIsDone(t *testing.T) {
	toks := tokensOf("{", "\"n\"", ":", " ", "12", "}")
	g := New(mustParse(t, objectGrammar), toks)
	eos := idOf(toks, "<eos>")

	if g.Allows(eos) {
		t.Fatal("the turn could end before the object opened")
	}
	feed(t, g, toks, "{", "\"n\"", ":", "12")
	if g.Allows(eos) {
		t.Fatal("the turn could end with the object still open")
	}
	feed(t, g, toks, "}")
	if !g.Allows(eos) {
		t.Fatal("the object closed and the turn still cannot end")
	}
}

func TestATokenThatPrintsNothingIsRefused(t *testing.T) {
	toks := tokensOf("{")
	g := New(mustParse(t, objectGrammar), toks)
	if g.Allows(0) {
		t.Fatal("a token with no piece was allowed")
	}
}

// The guard under the slow path: a state that refuses nearly everything says so
// in one table, and a sweep tests a byte before it walks a stack.
func TestFirstBytesNarrowsTheRow(t *testing.T) {
	toks := tokensOf("{", "\"n\"", ":", " ", "12", "}")
	g := New(mustParse(t, objectGrammar), toks)

	first := g.FirstBytes()
	if !first['{'] {
		t.Fatal("an object may start with a brace and the table says otherwise")
	}
	for _, b := range []byte{'"', 'x', '0', '}'} {
		if first[b] {
			t.Fatalf("the table allows %q where only a brace fits", b)
		}
	}

	feed(t, g, toks, "{")
	if next := g.FirstBytes(); !next['"'] && !next[' '] {
		t.Fatal("after a brace the grammar wants space or a quote, and the table allows neither")
	}
}

// The table may never refuse a token the grammar would have allowed: it is a
// guard in front of the walk, not a second opinion about it.
func TestFirstBytesNeverRefusesAnAllowedToken(t *testing.T) {
	toks := tokensOf("{", "\"n\"", ":", " ", "\t", "12", "7", "}", "x", "é", "\xc3")
	g := New(mustParse(t, objectGrammar), toks)

	for _, piece := range []string{"{", "\"n\"", ":", " ", "12", "}"} {
		first := g.FirstBytes()
		for id := range toks.bytes {
			if len(toks.bytes[id]) == 0 || toks.eog[id] {
				continue
			}
			if g.Allows(int32(id)) && !first[toks.first[id]] {
				t.Fatalf("%q is allowed and its first byte is not in the table", toks.bytes[id])
			}
		}
		feed(t, g, toks, piece)
	}
}

// A rule that refers to itself is how a repetition is written, and the stack
// set has to stay finite while it does.
func TestARecursiveRuleTerminates(t *testing.T) {
	toks := tokensOf("a", "b")
	g := New(mustParse(t, "root ::= a\na ::= \"a\" a | \"b\""), toks)
	feed(t, g, toks, "a", "a", "a", "b")
	if !g.Done() {
		t.Fatal("the recursion closed on b and the grammar is not done")
	}
}

func TestAGrammarWithoutARootIsRefused(t *testing.T) {
	if _, err := Parse(`other ::= "a"`); err == nil || !strings.Contains(err.Error(), "root") {
		t.Fatalf("a grammar with no root was accepted: %v", err)
	}
}
