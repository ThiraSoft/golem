// Package grammar constrains what a model may say next.
//
// A grammar is written in GBNF, llama.cpp's notation, and this is a port of
// its implementation (src/llama-grammar.cpp): the same elements, the same
// generated rule names, the same pushdown automaton over code points. A schema
// compiled by grammar/schema arrives here as GBNF text, so there is one engine
// and not two.
//
// What a grammar knows how to answer is whether a token may come next. Nothing
// here reads logits or draws anything; sample.Sampler holds one of these and
// asks.
package grammar

import "fmt"

// Kind is what an element of a rule is. The two token elements llama.cpp also
// carries — <[id]> and its negation — are not here: nothing golem produces
// writes them, and they are the only part of a grammar that depends on which
// vocabulary it was written for.
type Kind uint8

const (
	End          Kind = iota // the rule ends here
	Alt                      // the alternatives of a rule are separated by this
	RuleRef                  // Value is the identifier of another rule
	Char                     // Value is a code point that matches
	CharNot                  // Value is a code point that must not match
	CharRngUpper             // upper bound of a range opened by the element before
	CharAlt                  // one more code point accepted by that same element
	CharAny                  // any code point at all
)

func (k Kind) String() string {
	switch k {
	case End:
		return "END"
	case Alt:
		return "ALT"
	case RuleRef:
		return "RULE_REF"
	case Char:
		return "CHAR"
	case CharNot:
		return "CHAR_NOT"
	case CharRngUpper:
		return "CHAR_RNG_UPPER"
	case CharAlt:
		return "CHAR_ALT"
	case CharAny:
		return "CHAR_ANY"
	}
	return fmt.Sprintf("Kind(%d)", uint8(k))
}

// Element is one step of a rule: a code point to match, a rule to descend into,
// or a mark.
type Element struct {
	Kind  Kind
	Value rune // a code point, or a rule identifier for RuleRef
}

// Rules is a parsed grammar: rules flattened into slices of elements, with the
// alternatives of a rule separated by Alt and the whole closed by End.
//
// A position inside one of them is two integers rather than the pointer
// llama.cpp carries, which is what makes a copy of the automaton's state cheap
// — and copying it is what Grammar.Allows does for every candidate it tests.
type Rules struct {
	rules [][]Element
	ids   map[string]uint32
	names []string
	root  uint32
}

// Rule returns the elements of a rule, which is what the automaton walks.
func (r *Rules) Rule(id uint32) []Element { return r.rules[id] }

// Root is the rule a grammar starts at.
func (r *Rules) Root() uint32 { return r.root }

// Name is what a rule was called in the source, for an error worth reading.
func (r *Rules) Name(id uint32) string {
	if int(id) < len(r.names) {
		return r.names[id]
	}
	return fmt.Sprintf("rule %d", id)
}
