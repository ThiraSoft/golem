package grammar

import (
	"strings"
	"testing"
)

// The cases are llama.cpp's own (tests/test-grammar-parser.cpp), because the
// identifiers, the generated names and the order of the elements are what
// grammar/schema's expected strings rest on.

func elems(t *testing.T, r *Rules, name string) []Element {
	t.Helper()
	id, ok := r.ids[name]
	if !ok {
		t.Fatalf("no rule named %q; the grammar declares %v", name, r.names)
	}
	return r.rules[id]
}

func want(t *testing.T, got []Element, expected ...Element) {
	t.Helper()
	if len(got) != len(expected) {
		t.Fatalf("the rule holds %d elements and %d were expected:\n%v", len(got), len(expected), got)
	}
	for i := range got {
		if got[i] != expected[i] {
			t.Fatalf("element %d is %v %q, expected %v %q", i,
				got[i].Kind, got[i].Value, expected[i].Kind, expected[i].Value)
		}
	}
}

func TestALiteralIsASequenceOfCharacters(t *testing.T) {
	r, err := Parse(`root ::= "a"`)
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"), Element{Char, 'a'}, Element{End, 0})
}

func TestAlternativesRangesAndNegation(t *testing.T) {
	r, err := Parse(`root ::= "a" | [bdx-z] | [^1-3]`)
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"),
		Element{Char, 'a'},
		Element{Alt, 0},
		Element{Char, 'b'},
		Element{CharAlt, 'd'},
		Element{CharAlt, 'x'},
		Element{CharRngUpper, 'z'},
		Element{Alt, 0},
		Element{CharNot, '1'},
		Element{CharRngUpper, '3'},
		Element{End, 0})
}

// A repetition is rewritten into a rule that refers to itself, and the rule it
// generates is named after the one it came from plus its own identifier.
func TestPlusRewritesIntoARecursiveRule(t *testing.T) {
	r, err := Parse("root ::= a+\na ::= \"a\"")
	if err != nil {
		t.Fatal(err)
	}
	if r.ids["root"] != 0 || r.ids["a"] != 1 || r.ids["root_2"] != 2 {
		t.Fatalf("the identifiers are %v, and llama.cpp gives root 0, a 1, root_2 2", r.ids)
	}
	want(t, elems(t, r, "root"),
		Element{RuleRef, 1}, Element{RuleRef, 2}, Element{End, 0})
	want(t, elems(t, r, "root_2"),
		Element{RuleRef, 1}, Element{RuleRef, 2}, Element{Alt, 0}, Element{End, 0})
}

func TestQuestionMarkRewritesIntoAnEmptyAlternative(t *testing.T) {
	r, err := Parse(`root ::= "a"?`)
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"), Element{RuleRef, 1}, Element{End, 0})
	want(t, elems(t, r, "root_1"), Element{Char, 'a'}, Element{Alt, 0}, Element{End, 0})
}

func TestStarRewritesIntoARuleThatMayMatchNothing(t *testing.T) {
	r, err := Parse(`root ::= "a"*`)
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"), Element{RuleRef, 1}, Element{End, 0})
	want(t, elems(t, r, "root_1"),
		Element{Char, 'a'}, Element{RuleRef, 1}, Element{Alt, 0}, Element{End, 0})
}

func TestABoundedRepetitionRepeatsAndThenOffers(t *testing.T) {
	r, err := Parse(`root ::= "a"{2,3}`)
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"),
		Element{Char, 'a'}, Element{Char, 'a'}, Element{RuleRef, 1}, Element{End, 0})
	want(t, elems(t, r, "root_1"), Element{Char, 'a'}, Element{Alt, 0}, Element{End, 0})
}

func TestAnExactRepetitionIsJustTheItemTwice(t *testing.T) {
	r, err := Parse(`root ::= "a"{2}`)
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"), Element{Char, 'a'}, Element{Char, 'a'}, Element{End, 0})
}

func TestAGroupBecomesARuleOfItsOwn(t *testing.T) {
	r, err := Parse(`root ::= ("a" | "b") "c"`)
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"), Element{RuleRef, 1}, Element{Char, 'c'}, Element{End, 0})
	want(t, elems(t, r, "root_1"),
		Element{Char, 'a'}, Element{Alt, 0}, Element{Char, 'b'}, Element{End, 0})
}

func TestEscapesAndTheDot(t *testing.T) {
	r, err := Parse(`root ::= "\n\t\\\"" [\x41-\x43] . "é"`)
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"),
		Element{Char, '\n'}, Element{Char, '\t'}, Element{Char, '\\'}, Element{Char, '"'},
		Element{Char, 'A'}, Element{CharRngUpper, 'C'},
		Element{CharAny, 0},
		Element{Char, 'é'},
		Element{End, 0})
}

func TestCommentsAndBlankLinesAreSkipped(t *testing.T) {
	r, err := Parse("# what this grammar is for\nroot ::= \"a\" # and this rule\n\n")
	if err != nil {
		t.Fatal(err)
	}
	want(t, elems(t, r, "root"), Element{Char, 'a'}, Element{End, 0})
}

// A grammar that does not parse says where it stopped, because a grammar is
// usually generated and the line is the only clue about which part.
func TestWhatIsRefused(t *testing.T) {
	for _, c := range []struct{ src, says string }{
		{`root ::= "a"{,}`, "expecting a number"},
		{`root ::= "a"{,10}`, "expecting a number"},
		{`root ::= *`, "nothing in front of it"},
		{`root ::= "a`, "never closed"},
		{`root ::= [a`, "never closed"},
		{`root ::= a`, "never defined"},
		{`nothing ::= "a"`, "no rule named root"},
		{`root ::= ("a"`, "expecting )"},
		{`root "a"`, "expecting ::="},
		{`root ::= "\q"`, "escape"},
		{`root ::= (((((([^x]*){0,99}){0,99}){0,99}){0,99}){0,99}){0,99}`, "wider than"},
	} {
		_, err := Parse(c.src)
		if err == nil {
			t.Fatalf("%q parsed, and should not have", c.src)
		}
		if !strings.Contains(err.Error(), c.says) {
			t.Fatalf("%q was refused with %q, which does not say %q", c.src, err, c.says)
		}
	}
}
