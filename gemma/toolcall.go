package gemma

// Reading back what the model wrote.
//
// The arguments of a call are not JSON: <|"|> is the quote, keys are bare, and
// a string may hold a brace or a comma. So this is a scanner rather than a call
// to encoding/json — and it refuses what it cannot read rather than returning
// half a call, because a half-read call would be executed.

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseToolCalls splits an answer into the prose that came before the first
// call and the calls that followed.
func ParseToolCalls(text string) (string, []ToolCall, error) {
	head := strings.Index(text, toolCallOpen)
	if head < 0 {
		return text, nil, nil
	}
	before := text[:head]
	rest := text[head:]

	var calls []ToolCall
	for strings.HasPrefix(rest, toolCallOpen) {
		rest = rest[len(toolCallOpen):]
		if !strings.HasPrefix(rest, "call:") {
			return before, nil, fmt.Errorf("gemma: a tool call that does not open with call:")
		}
		rest = rest[len("call:"):]
		brace := strings.Index(rest, "{")
		if brace < 0 {
			return before, nil, fmt.Errorf("gemma: a tool call with no arguments block")
		}
		name := strings.TrimSpace(rest[:brace])
		if name == "" {
			return before, nil, fmt.Errorf("gemma: a tool call with no function name")
		}
		p := &argScanner{s: rest[brace:]}
		args, err := p.mapping()
		if err != nil {
			return before, nil, err
		}
		rest = p.s[p.at:]
		if !strings.HasPrefix(rest, toolCallClose) {
			return before, nil, fmt.Errorf("gemma: a tool call that never closes")
		}
		rest = rest[len(toolCallClose):]
		calls = append(calls, ToolCall{Name: name, Arguments: args})
	}
	return before, calls, nil
}

// argScanner walks one argument block.
type argScanner struct {
	s  string
	at int
}

func (p *argScanner) mapping() (map[string]any, error) {
	if !p.take("{") {
		return nil, fmt.Errorf("gemma: expected { at %d", p.at)
	}
	out := map[string]any{}
	if p.take("}") {
		return out, nil
	}
	for {
		key, err := p.key()
		if err != nil {
			return nil, err
		}
		if !p.take(":") {
			return nil, fmt.Errorf("gemma: the argument %q has no value", key)
		}
		value, err := p.value()
		if err != nil {
			return nil, err
		}
		out[key] = value
		if p.take(",") {
			continue
		}
		if p.take("}") {
			return out, nil
		}
		return nil, fmt.Errorf("gemma: expected , or } at %d", p.at)
	}
}

// key reads a bare key, up to its colon.
func (p *argScanner) key() (string, error) {
	start := p.at
	for p.at < len(p.s) && p.s[p.at] != ':' && p.s[p.at] != ',' && p.s[p.at] != '}' {
		p.at++
	}
	key := strings.TrimSpace(p.s[start:p.at])
	if key == "" {
		return "", fmt.Errorf("gemma: an argument with no key at %d", start)
	}
	return key, nil
}

func (p *argScanner) value() (any, error) {
	switch {
	case p.peek(quote):
		return p.text()
	case p.peek("{"):
		return p.mapping()
	case p.peek("["):
		return p.list()
	default:
		return p.scalar()
	}
}

// text reads a quoted string: everything up to the closing <|"|>, whatever it
// holds.
func (p *argScanner) text() (string, error) {
	p.take(quote)
	end := strings.Index(p.s[p.at:], quote)
	if end < 0 {
		return "", fmt.Errorf("gemma: a string that is never closed")
	}
	out := p.s[p.at : p.at+end]
	p.at += end + len(quote)
	return out, nil
}

func (p *argScanner) list() ([]any, error) {
	p.take("[")
	out := []any{}
	if p.take("]") {
		return out, nil
	}
	for {
		v, err := p.value()
		if err != nil {
			return nil, err
		}
		out = append(out, v)
		if p.take(",") {
			continue
		}
		if p.take("]") {
			return out, nil
		}
		return nil, fmt.Errorf("gemma: expected , or ] at %d", p.at)
	}
}

// scalar reads a bare word: a number, a boolean, or nothing at all.
func (p *argScanner) scalar() (any, error) {
	start := p.at
	for p.at < len(p.s) && p.s[p.at] != ',' && p.s[p.at] != '}' && p.s[p.at] != ']' {
		p.at++
	}
	word := strings.TrimSpace(p.s[start:p.at])
	switch word {
	case "true":
		return true, nil
	case "false":
		return false, nil
	case "None", "null":
		return nil, nil
	case "":
		return nil, fmt.Errorf("gemma: an argument with no value at %d", start)
	}
	if f, err := strconv.ParseFloat(word, 64); err == nil {
		return f, nil
	}
	// A bare word that is neither a number nor a keyword: the model wrote an
	// unquoted string, and dropping it would lose an argument.
	return word, nil
}

func (p *argScanner) peek(what string) bool {
	return strings.HasPrefix(p.s[p.at:], what)
}

func (p *argScanner) take(what string) bool {
	if p.peek(what) {
		p.at += len(what)
		return true
	}
	return false
}
