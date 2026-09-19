package jinja

// Cutting a template into text and tags, with the whitespace rules
// transformers renders with.
//
// transformers opens its environment with trim_blocks and lstrip_blocks, and
// every chat template is written against those two: the first newline after a
// block or comment tag is dropped, and the spaces and tabs before one are
// dropped when nothing else stands between it and the start of its line. On
// top of that a tag may carry a minus on either side, which eats every blank
// on that side, newlines included, and a plus on its left, which keeps the
// indentation lstrip_blocks would have taken.

import (
	"fmt"
	"strconv"
	"strings"
)

type chunkKind int

const (
	chunkText  chunkKind = iota
	chunkVar             // {{ }}
	chunkBlock           // {% %}
)

type chunk struct {
	kind chunkKind
	text string  // chunkText
	toks []token // chunkVar and chunkBlock
	line int
}

type tokKind int

const (
	tokName tokKind = iota
	tokString
	tokInt
	tokFloat
	tokOp
	tokEOF
)

type token struct {
	kind tokKind
	s    string // the name, the decoded string, the operator
	i    int64
	f    float64
	line int
}

func (t token) String() string {
	switch t.kind {
	case tokString:
		return reprString(t.s)
	case tokInt:
		return fmt.Sprint(t.i)
	case tokFloat:
		return formatFloat(t.f)
	case tokEOF:
		return "end of tag"
	}
	return t.s
}

type lexer struct {
	src  string
	pos  int
	line int
	out  []chunk
	// lineStart says the last tag ended at the start of a line, which is what
	// lets lstrip_blocks act on text holding no newline of its own.
	lineStart bool
}

func lex(src string) ([]chunk, error) {
	l := &lexer{src: src, line: 1, lineStart: true}
	for l.pos < len(src) {
		open := nextOpen(src, l.pos)
		if open < 0 {
			l.text(src[l.pos:])
			break
		}
		text := src[l.pos:open]
		kind := src[open+1]
		sign := byte(0)
		if open+2 < len(src) && (src[open+2] == '-' || src[open+2] == '+') {
			sign = src[open+2]
		}
		switch {
		case sign == '-':
			text = strings.TrimRight(text, " \t\n\r\f\v")
		case sign != '+' && kind != '{':
			// lstrip_blocks: blanks from the start of the line to the tag go.
			at := strings.LastIndexByte(text, '\n') + 1
			if (at > 0 || l.lineStart) && strings.Trim(text[at:], " \t") == "" {
				text = text[:at]
			}
		}
		l.text(text)
		l.line += strings.Count(src[l.pos:open], "\n")
		l.pos = open + 2
		if sign != 0 {
			l.pos++
		}
		var err error
		switch kind {
		case '#':
			err = l.comment()
		case '{':
			err = l.tag(chunkVar)
		case '%':
			err = l.tag(chunkBlock)
			if err == nil {
				err = l.raw()
			}
		}
		if err != nil {
			return nil, err
		}
	}
	return l.out, nil
}

// nextOpen finds the next {{, {% or {#.
func nextOpen(src string, from int) int {
	for i := from; i+1 < len(src); i++ {
		if src[i] == '{' && (src[i+1] == '{' || src[i+1] == '%' || src[i+1] == '#') {
			return i
		}
	}
	return -1
}

func (l *lexer) text(s string) {
	if s != "" {
		l.out = append(l.out, chunk{kind: chunkText, text: s, line: l.line})
	}
}

// closeTag handles what follows a tag: a minus ate its right side already;
// otherwise a block or comment drops the one newline right after it.
func (l *lexer) closeTag(minus bool, block bool) {
	start := l.pos
	switch {
	case minus:
		for l.pos < len(l.src) && strings.IndexByte(" \t\n\r\f\v", l.src[l.pos]) >= 0 {
			l.pos++
		}
	case block && l.pos < len(l.src) && l.src[l.pos] == '\n':
		l.pos++
	}
	l.line += strings.Count(l.src[start:l.pos], "\n")
	l.lineStart = l.pos > 0 && l.src[l.pos-1] == '\n'
}

func (l *lexer) comment() error {
	end := strings.Index(l.src[l.pos:], "#}")
	if end < 0 {
		return fmt.Errorf("line %d: a comment is never closed", l.line)
	}
	body := l.src[l.pos : l.pos+end]
	l.line += strings.Count(body, "\n")
	l.pos += end + 2
	l.closeTag(strings.HasSuffix(body, "-"), true)
	return nil
}

// raw copies what sits between {% raw %} and {% endraw %} as text.
func (l *lexer) raw() error {
	last := l.out[len(l.out)-1]
	if len(last.toks) != 2 || last.toks[0].kind != tokName || last.toks[0].s != "raw" {
		return nil
	}
	l.out = l.out[:len(l.out)-1]
	for at := l.pos; ; {
		open := strings.Index(l.src[at:], "{%")
		if open < 0 {
			return fmt.Errorf("line %d: a raw block is never closed", last.line)
		}
		open += at
		i := open + 2
		minus := i < len(l.src) && l.src[i] == '-'
		if minus || (i < len(l.src) && l.src[i] == '+') {
			i++
		}
		rest := strings.TrimLeft(l.src[i:], " \t\n")
		if !strings.HasPrefix(rest, "endraw") {
			at = open + 2
			continue
		}
		text := l.src[l.pos:open]
		if minus {
			text = strings.TrimRight(text, " \t\n\r\f\v")
		}
		l.text(text)
		l.line += strings.Count(l.src[l.pos:open], "\n")
		l.pos = i
		return l.tag(-1)
	}
}

// tag reads the tokens of one {{ }} or {% %}, up to its closing marker. A
// kind of -1 reads an {% endraw %} and keeps nothing.
func (l *lexer) tag(kind chunkKind) error {
	closer := "%}"
	if kind == chunkVar {
		closer = "}}"
	}
	line := l.line
	var toks []token
	depth := 0
	for {
		for l.pos < len(l.src) && strings.IndexByte(" \t\n\r", l.src[l.pos]) >= 0 {
			if l.src[l.pos] == '\n' {
				l.line++
			}
			l.pos++
		}
		if l.pos >= len(l.src) {
			return fmt.Errorf("line %d: a tag is never closed", line)
		}
		rest := l.src[l.pos:]
		if depth == 0 {
			minus := strings.HasPrefix(rest, "-"+closer)
			if minus || strings.HasPrefix(rest, closer) || strings.HasPrefix(rest, "+"+closer) {
				l.pos += len(closer)
				if minus || rest[0] == '+' {
					l.pos++
				}
				l.closeTag(minus, kind != chunkVar && rest[0] != '+')
				if kind >= 0 {
					toks = append(toks, token{kind: tokEOF, line: l.line})
					l.out = append(l.out, chunk{kind: kind, toks: toks, line: line})
				}
				return nil
			}
		}
		t, err := l.token()
		if err != nil {
			return err
		}
		if t.kind == tokOp {
			switch t.s {
			case "(", "[", "{":
				depth++
			case ")", "]", "}":
				depth--
			}
		}
		toks = append(toks, t)
	}
}

// Operators, the two-character ones first so they win.
var operators = []string{"//", "**", "==", "!=", "<=", ">=",
	"+", "-", "*", "/", "%", "~", "<", ">", "=", "(", ")", "[", "]", "{", "}", ",", ".", ":", "|"}

func (l *lexer) token() (token, error) {
	src, start := l.src, l.pos
	c := src[start]
	switch {
	case c == '_' || isLetter(c):
		i := start
		for i < len(src) && (src[i] == '_' || isLetter(src[i]) || isDigit(src[i])) {
			i++
		}
		l.pos = i
		return token{kind: tokName, s: src[start:i], line: l.line}, nil
	case isDigit(c):
		return l.number()
	case c == '\'' || c == '"':
		return l.string(c)
	}
	for _, op := range operators {
		if strings.HasPrefix(src[start:], op) {
			l.pos += len(op)
			return token{kind: tokOp, s: op, line: l.line}, nil
		}
	}
	return token{}, fmt.Errorf("line %d: unexpected character %q", l.line, c)
}

func isLetter(c byte) bool { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' }
func isDigit(c byte) bool  { return c >= '0' && c <= '9' }

func (l *lexer) number() (token, error) {
	src, i := l.src, l.pos
	float := false
	for i < len(src) && (isDigit(src[i]) || src[i] == '_') {
		i++
	}
	if i+1 < len(src) && src[i] == '.' && isDigit(src[i+1]) {
		float = true
		i++
		for i < len(src) && (isDigit(src[i]) || src[i] == '_') {
			i++
		}
	}
	if i < len(src) && (src[i] == 'e' || src[i] == 'E') {
		j := i + 1
		if j < len(src) && (src[j] == '+' || src[j] == '-') {
			j++
		}
		if j < len(src) && isDigit(src[j]) {
			float = true
			for i = j; i < len(src) && isDigit(src[i]); i++ {
			}
		}
	}
	text := strings.ReplaceAll(src[l.pos:i], "_", "")
	l.pos = i
	if float {
		f, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return token{}, fmt.Errorf("line %d: %q is not a number", l.line, text)
		}
		return token{kind: tokFloat, f: f, line: l.line}, nil
	}
	n, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		return token{}, fmt.Errorf("line %d: %q is not a number", l.line, text)
	}
	return token{kind: tokInt, i: n, line: l.line}, nil
}

// string reads a quoted literal and decodes Python's escapes, which Jinja
// applies to every string in a tag.
func (l *lexer) string(q byte) (token, error) {
	src, i := l.src, l.pos+1
	line := l.line
	var b strings.Builder
	for {
		if i >= len(src) {
			return token{}, fmt.Errorf("line %d: a string is never closed", line)
		}
		c := src[i]
		if c == q {
			l.pos = i + 1
			return token{kind: tokString, s: b.String(), line: line}, nil
		}
		if c == '\n' {
			l.line++
		}
		if c != '\\' || i+1 >= len(src) {
			b.WriteByte(c)
			i++
			continue
		}
		e := src[i+1]
		i += 2
		switch e {
		case 'n':
			b.WriteByte('\n')
		case 't':
			b.WriteByte('\t')
		case 'r':
			b.WriteByte('\r')
		case '0':
			b.WriteByte(0)
		case 'a':
			b.WriteByte('\a')
		case 'b':
			b.WriteByte('\b')
		case 'f':
			b.WriteByte('\f')
		case 'v':
			b.WriteByte('\v')
		case '\\', '\'', '"':
			b.WriteByte(e)
		case '\n':
			l.line++ // an escaped newline continues the line
		case 'x', 'u', 'U':
			n := map[byte]int{'x': 2, 'u': 4, 'U': 8}[e]
			if i+n > len(src) {
				return token{}, fmt.Errorf("line %d: a truncated \\%c escape", l.line, e)
			}
			r, err := strconv.ParseUint(src[i:i+n], 16, 32)
			if err != nil {
				return token{}, fmt.Errorf("line %d: a bad \\%c escape", l.line, e)
			}
			b.WriteRune(rune(r))
			i += n
		default:
			// Python keeps an unknown escape as it was written.
			b.WriteByte('\\')
			b.WriteByte(e)
		}
	}
}
