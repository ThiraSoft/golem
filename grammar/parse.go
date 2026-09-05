package grammar

// The GBNF parser, ported from llama_grammar_parser (llama-grammar.cpp:413-716).
//
// Fidelity matters more than elegance here: the identifiers rules are given,
// the names generated for a repetition or a group, and the order elements are
// emitted in are all observable, and grammar/schema's tests compare what golem
// produces against what llama.cpp produces.

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// maxRepetition is llama.cpp's MAX_REPETITION_THRESHOLD (llama-grammar.cpp:13):
// a bound on what a rewritten {m,n} may cost, since the rewrite is a rule per
// repetition and a grammar can ask for a great many.
const maxRepetition = 2000

// Parse reads GBNF and returns the rules it declares. The grammar must declare
// a rule named root, which is where a walk starts.
func Parse(src string) (*Rules, error) {
	p := &parser{src: src, r: &Rules{ids: map[string]uint32{}}}
	if err := p.parse(); err != nil {
		return nil, err
	}
	return p.r, nil
}

type parser struct {
	src string
	pos int
	r   *Rules
}

func (p *parser) parse() error {
	p.space(true)
	for p.pos < len(p.src) {
		if err := p.rule(); err != nil {
			return err
		}
	}
	// Every rule referred to has to exist: a reference to a name nobody
	// defined parses cleanly and then walks into an empty rule.
	for id, rule := range p.r.rules {
		if len(rule) == 0 {
			return fmt.Errorf("grammar: rule %q is referred to and never defined", p.r.Name(uint32(id)))
		}
		for _, e := range rule {
			if e.Kind != RuleRef {
				continue
			}
			if int(e.Value) >= len(p.r.rules) || len(p.r.rules[e.Value]) == 0 {
				return fmt.Errorf("grammar: rule %q refers to %q, which is never defined",
					p.r.Name(uint32(id)), p.r.Name(uint32(e.Value)))
			}
		}
	}
	root, ok := p.r.ids["root"]
	if !ok {
		return fmt.Errorf("grammar: no rule named root")
	}
	p.r.root = root
	return nil
}

// rule reads one `name ::= alternates` and the newline that ends it.
func (p *parser) rule() error {
	name, err := p.name()
	if err != nil {
		return err
	}
	id := p.symbolID(name)
	p.space(false)
	if !strings.HasPrefix(p.src[p.pos:], "::=") {
		return p.errorf("expecting ::=")
	}
	p.pos += 3
	p.space(true)
	if err := p.alternates(name, id, false); err != nil {
		return err
	}
	switch {
	case p.peek() == '\r':
		p.pos++
		if p.peek() == '\n' {
			p.pos++
		}
	case p.peek() == '\n':
		p.pos++
	case p.pos < len(p.src):
		return p.errorf("expecting a newline or the end of the grammar")
	}
	p.space(true)
	return nil
}

func (p *parser) alternates(name string, id uint32, nested bool) error {
	var rule []Element
	if err := p.sequence(name, &rule, nested); err != nil {
		return err
	}
	for p.peek() == '|' {
		rule = append(rule, Element{Kind: Alt})
		p.pos++
		p.space(true)
		if err := p.sequence(name, &rule, nested); err != nil {
			return err
		}
	}
	rule = append(rule, Element{Kind: End})
	p.addRule(id, rule)
	return nil
}

// sequence reads one alternative: literals, character classes, references,
// groups and the repetition operators that rewrite what came before them.
func (p *parser) sequence(name string, rule *[]Element, nested bool) error {
	lastSym := len(*rule)
	prevRules := uint64(1)

	for p.pos < len(p.src) {
		switch c := p.peek(); {
		case c == '"':
			p.pos++
			lastSym, prevRules = len(*rule), 1
			for p.peek() != '"' {
				if p.pos >= len(p.src) {
					return p.errorf("a string that is never closed")
				}
				r, err := p.char()
				if err != nil {
					return err
				}
				*rule = append(*rule, Element{Kind: Char, Value: r})
			}
			p.pos++
			p.space(nested)

		case c == '[':
			p.pos++
			start := Char
			if p.peek() == '^' {
				p.pos++
				start = CharNot
			}
			lastSym, prevRules = len(*rule), 1
			for p.peek() != ']' {
				if p.pos >= len(p.src) {
					return p.errorf("a character class that is never closed")
				}
				r, err := p.char()
				if err != nil {
					return err
				}
				kind := start
				if lastSym < len(*rule) {
					kind = CharAlt
				}
				*rule = append(*rule, Element{Kind: kind, Value: r})
				// A dash that is not the last character opens a range.
				if p.peek() == '-' && p.peekAt(1) != ']' {
					p.pos++
					if p.pos >= len(p.src) {
						return p.errorf("a range with no upper bound")
					}
					upper, err := p.char()
					if err != nil {
						return err
					}
					*rule = append(*rule, Element{Kind: CharRngUpper, Value: upper})
				}
			}
			p.pos++
			p.space(nested)

		case c == '(':
			p.pos++
			p.space(true)
			before := len(p.r.ids)
			sub := p.generateSymbolID(name)
			if err := p.alternates(name, sub, true); err != nil {
				return err
			}
			if n := uint64(len(p.r.ids) - before); n > 1 {
				prevRules = n
			} else {
				prevRules = 1
			}
			lastSym = len(*rule)
			*rule = append(*rule, Element{Kind: RuleRef, Value: rune(sub)})
			if p.peek() != ')' {
				return p.errorf("expecting )")
			}
			p.pos++
			p.space(nested)

		case c == '.':
			lastSym, prevRules = len(*rule), 1
			*rule = append(*rule, Element{Kind: CharAny})
			p.pos++
			p.space(nested)

		case c == '*' || c == '+' || c == '?':
			p.pos++
			p.space(nested)
			min, max := uint64(0), noMax
			if c == '+' {
				min = 1
			}
			if c == '?' {
				max = 1
			}
			if err := p.repeat(name, rule, lastSym, &prevRules, min, max); err != nil {
				return err
			}

		case c == '{':
			p.pos++
			p.space(nested)
			min, err := p.integer()
			if err != nil {
				return err
			}
			p.space(nested)
			max := noMax
			switch p.peek() {
			case '}':
				max = min
				p.pos++
				p.space(nested)
			case ',':
				p.pos++
				p.space(nested)
				if isDigit(p.peek()) {
					if max, err = p.integer(); err != nil {
						return err
					}
					p.space(nested)
				}
				if p.peek() != '}' {
					return p.errorf("expecting }")
				}
				p.pos++
				p.space(nested)
			default:
				return p.errorf("expecting , or }")
			}
			if min > maxRepetition || (max != noMax && max > maxRepetition) {
				return p.errorf("more repetitions than a grammar has any use for")
			}
			if err := p.repeat(name, rule, lastSym, &prevRules, min, max); err != nil {
				return err
			}

		case isWordChar(c):
			word, err := p.name()
			if err != nil {
				return err
			}
			ref := p.symbolID(word)
			p.space(nested)
			lastSym, prevRules = len(*rule), 1
			*rule = append(*rule, Element{Kind: RuleRef, Value: rune(ref)})

		default:
			return nil
		}
	}
	return nil
}

// noMax is "no upper bound", the value llama.cpp writes as UINT64_MAX.
const noMax = ^uint64(0)

// repeat rewrites the item just read as a repetition of itself, exactly as
// llama-grammar.cpp:455-521 does:
//
//	S{m,n} --> S S S (m times) S'(n-m), S'(x) ::= S S'(x-1) |
//	S{m,}  --> S S S (m times) S',      S'    ::= S S' |
//	S*     --> S{0,}
//	S+     --> S{1,}
//	S?     --> S{0,1}
func (p *parser) repeat(name string, rule *[]Element, lastSym int, prevRules *uint64, min, max uint64) error {
	if lastSym == len(*rule) {
		return p.errorf("a repetition with nothing in front of it")
	}
	prev := append([]Element(nil), (*rule)[lastSym:]...)

	total := uint64(1)
	switch {
	case max != noMax && max > 0:
		total = max
	case min > 0:
		total = min
	}
	if *prevRules*total >= maxRepetition {
		return p.errorf("a repetition of a repetition, wider than a grammar has any use for")
	}

	if min == 0 {
		*rule = (*rule)[:lastSym]
	} else {
		for i := uint64(1); i < min; i++ {
			*rule = append(*rule, prev...)
		}
	}

	opts := uint64(1)
	if max != noMax {
		opts = max - min
	}
	var last uint32
	for i := uint64(0); i < opts; i++ {
		rec := append([]Element(nil), prev...)
		id := p.generateSymbolID(name)
		if i > 0 || max == noMax {
			ref := id
			if max != noMax {
				ref = last
			}
			rec = append(rec, Element{Kind: RuleRef, Value: rune(ref)})
		}
		rec = append(rec, Element{Kind: Alt}, Element{Kind: End})
		p.addRule(id, rec)
		last = id
	}
	if opts > 0 {
		*rule = append(*rule, Element{Kind: RuleRef, Value: rune(last)})
	}
	*prevRules *= total
	return nil
}

func (p *parser) symbolID(name string) uint32 {
	if id, ok := p.r.ids[name]; ok {
		return id
	}
	id := uint32(len(p.r.ids))
	p.r.ids[name] = id
	p.r.remember(id, name)
	return id
}

// generateSymbolID names a rule the source did not: a group, or the tail of a
// repetition. The name is the enclosing rule's plus the identifier, which is
// what llama.cpp does and what makes the two produce the same text.
func (p *parser) generateSymbolID(base string) uint32 {
	id := uint32(len(p.r.ids))
	p.r.ids[fmt.Sprintf("%s_%d", base, id)] = id
	p.r.remember(id, fmt.Sprintf("%s_%d", base, id))
	return id
}

func (r *Rules) remember(id uint32, name string) {
	for len(r.names) <= int(id) {
		r.names = append(r.names, "")
	}
	r.names[id] = name
}

func (p *parser) addRule(id uint32, rule []Element) {
	for len(p.r.rules) <= int(id) {
		p.r.rules = append(p.r.rules, nil)
	}
	p.r.rules[id] = rule
}

// space skips blanks and comments. A comment runs to the end of its line, and
// whether a newline is itself space depends on where the parser is: inside a
// group it is, between two rules it ends one.
func (p *parser) space(newlineOK bool) {
	for p.pos < len(p.src) {
		switch c := p.src[p.pos]; {
		case c == ' ' || c == '\t':
			p.pos++
		case c == '#':
			for p.pos < len(p.src) && p.src[p.pos] != '\r' && p.src[p.pos] != '\n' {
				p.pos++
			}
		case newlineOK && (c == '\r' || c == '\n'):
			p.pos++
		default:
			return
		}
	}
}

func (p *parser) name() (string, error) {
	start := p.pos
	for p.pos < len(p.src) && isWordChar(p.src[p.pos]) {
		p.pos++
	}
	if p.pos == start {
		return "", p.errorf("expecting the name of a rule")
	}
	return p.src[start:p.pos], nil
}

func (p *parser) integer() (uint64, error) {
	start := p.pos
	var v uint64
	for p.pos < len(p.src) && isDigit(p.src[p.pos]) {
		v = v*10 + uint64(p.src[p.pos]-'0')
		if v > maxRepetition {
			// Reading further would only overflow; the caller refuses it.
			v = maxRepetition + 1
		}
		p.pos++
	}
	if p.pos == start {
		return 0, p.errorf("expecting a number")
	}
	return v, nil
}

// char reads one code point of a literal or a class, escapes included.
func (p *parser) char() (rune, error) {
	if p.pos >= len(p.src) {
		return 0, p.errorf("the grammar ends in the middle of a character")
	}
	if p.src[p.pos] != '\\' {
		r, size := utf8.DecodeRuneInString(p.src[p.pos:])
		if r == utf8.RuneError && size <= 1 {
			return 0, p.errorf("a byte that is not UTF-8")
		}
		p.pos += size
		return r, nil
	}
	if p.pos+1 >= len(p.src) {
		return 0, p.errorf("an escape with nothing after it")
	}
	esc := p.src[p.pos+1]
	p.pos += 2
	switch esc {
	case 'x':
		return p.hex(2)
	case 'u':
		return p.hex(4)
	case 'U':
		return p.hex(8)
	case 't':
		return '\t', nil
	case 'r':
		return '\r', nil
	case 'n':
		return '\n', nil
	case '\\', '"', '[', ']':
		return rune(esc), nil
	}
	p.pos -= 2
	return 0, p.errorf("an escape this grammar does not know")
}

func (p *parser) hex(n int) (rune, error) {
	if p.pos+n > len(p.src) {
		return 0, p.errorf(fmt.Sprintf("expecting %d hexadecimal digits", n))
	}
	var v rune
	for i := 0; i < n; i++ {
		c := p.src[p.pos+i]
		switch {
		case '0' <= c && c <= '9':
			v = v<<4 + rune(c-'0')
		case 'a' <= c && c <= 'f':
			v = v<<4 + rune(c-'a'+10)
		case 'A' <= c && c <= 'F':
			v = v<<4 + rune(c-'A'+10)
		default:
			return 0, p.errorf(fmt.Sprintf("expecting %d hexadecimal digits", n))
		}
	}
	p.pos += n
	return v, nil
}

func (p *parser) peek() byte { return p.peekAt(0) }

func (p *parser) peekAt(n int) byte {
	if p.pos+n >= len(p.src) {
		return 0
	}
	return p.src[p.pos+n]
}

// errorf names where in the source the parser gave up, because a grammar is
// usually generated and the line is the only clue about which part.
func (p *parser) errorf(what string) error {
	line, col := 1, 1
	for i := 0; i < p.pos && i < len(p.src); i++ {
		if p.src[i] == '\n' {
			line, col = line+1, 1
		} else {
			col++
		}
	}
	return fmt.Errorf("grammar: line %d, column %d: %s", line, col, what)
}

func isDigit(c byte) bool { return '0' <= c && c <= '9' }

func isWordChar(c byte) bool {
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || isDigit(c)
}
