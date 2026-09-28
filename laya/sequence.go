package laya

// What a question becomes before the model sees it: the checkpoint's
// render_options and build_sequence, and the state serialized as Python's
// json.dumps would, since the checkpoint was trained on those bytes and a
// different spacing is a different sequence.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// Question is one typed question, in the shape of a Jev request.
//
// Criteria is what the kind takes: for a choice, the options, either a list
// of labels or an object from label to description (null for none), in order;
// for a score, the list of levels from the lowest; for a noul, nothing, or an
// object whose "true" and "false" describe the two answers.
type Question struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

// question is a Question made ready: its kind, its instructions as one
// string, and its options as the model reads them with the keys the answer
// names them by.
type question struct {
	kind         int
	instructions string
	labels       []string // a choice's keys, a score's levels
	options      []string // the rendered texts, in label order
}

func prepare(q Question) (*question, error) {
	p := &question{kind: -1}
	for k, name := range kindNames {
		if q.Type == name {
			p.kind = k
		}
	}
	if p.kind < 0 {
		return nil, fmt.Errorf("type %q is none of choice, score and noul", q.Type)
	}
	if len(q.Instructions) == 0 {
		return nil, fmt.Errorf("the question has no instructions")
	}
	// A string is taken as it is; anything else as json.dumps writes it,
	// which escapes every character past ASCII.
	var s string
	if err := json.Unmarshal(q.Instructions, &s); err == nil {
		p.instructions = s
	} else {
		v, err := decodeOrdered(q.Instructions)
		if err != nil {
			return nil, fmt.Errorf("instructions: %w", err)
		}
		p.instructions = pyDumps(v, true)
	}

	crit, err := decodeOrdered(q.Criteria)
	if err != nil {
		return nil, fmt.Errorf("criteria: %w", err)
	}
	switch p.kind {
	case Choice:
		switch c := crit.(type) {
		case []any:
			for _, v := range c {
				label, ok := v.(string)
				if !ok {
					return nil, fmt.Errorf("a choice's options are strings")
				}
				p.labels = append(p.labels, label)
				p.options = append(p.options, label)
			}
		case *object:
			for i, k := range c.keys {
				p.labels = append(p.labels, k)
				desc := c.values[i]
				// k if not v else "k: v": null, "" and every other falsy
				// value leave the label alone.
				if falsy(desc) {
					p.options = append(p.options, k)
				} else {
					p.options = append(p.options, k+": "+pyStr(desc))
				}
			}
		default:
			return nil, fmt.Errorf("a choice's criteria are a list or an object")
		}
		if len(p.options) == 0 {
			return nil, fmt.Errorf("a choice needs options")
		}
	case Score:
		levels, ok := crit.([]any)
		if !ok || len(levels) == 0 {
			return nil, fmt.Errorf("a score's criteria are the list of its levels")
		}
		for i, v := range levels {
			p.labels = append(p.labels, strconv.Itoa(i))
			p.options = append(p.options, fmt.Sprintf("level %d: %s", i, pyStr(v)))
		}
	case Noul:
		desc := map[string]string{
			"false": "no, the statement does not hold",
			"true":  "yes, the statement holds",
		}
		if c, ok := crit.(*object); ok {
			for i, k := range c.keys {
				if _, known := desc[k]; known && !falsy(c.values[i]) {
					desc[k] = pyStr(c.values[i])
				}
			}
		} else if crit != nil {
			return nil, fmt.Errorf("a noul's criteria are an object with true and false")
		}
		p.options = []string{"false: " + desc["false"], "true: " + desc["true"]}
	}
	return p, nil
}

// sequence is build_sequence: the identifiers of one question's pass and
// where its option markers are.
func (m *Model) sequence(stateText string, q *question) ([]int32, []int) {
	cfg := m.Cfg
	enc := func(s string) []int32 {
		return m.Vocab.Encode(strings.ReplaceAll(s, m.Cfg.MaskText, " "), false, true)
	}
	head := enc(kindNames[q.kind] + " question: " + q.instructions)
	opts := make([][]int32, len(q.options))
	total := 0
	for i, o := range q.options {
		ids := enc(" " + o)
		opts[i] = append([]int32{cfg.Mask}, ids[:min(len(ids), 48)]...)
		total += len(opts[i])
	}
	budget := cfg.HeadMaxLen - total
	if budget < 16 {
		// Too many or too long: every option is cut to the same length.
		per := max(4, (cfg.HeadMaxLen-16)/max(1, len(opts)))
		total = 0
		for i := range opts {
			opts[i] = opts[i][:min(len(opts[i]), per)]
			total += len(opts[i])
		}
		budget = cfg.HeadMaxLen - total
	}
	head = head[:min(len(head), max(8, budget))]

	ids := append([]int32{cfg.CLS}, head...)
	ids = append(ids, cfg.SEP)
	var markers []int
	for _, o := range opts {
		markers = append(markers, len(ids))
		ids = append(ids, o...)
	}
	ids = append(ids, cfg.SEP)
	room := max(0, cfg.MaxLen-len(ids)-1)
	st := enc(stateText)
	ids = append(ids, st[:min(len(st), room)]...)
	ids = append(ids, cfg.SEP)
	ids = ids[:min(len(ids), cfg.MaxLen)]
	kept := markers[:0]
	for _, at := range markers {
		if at < cfg.MaxLen {
			kept = append(kept, at)
		}
	}
	return ids, kept
}

// StateText is a state as the model reads it: a JSON string is its own text,
// anything else is serialized as Python's json.dumps(ensure_ascii=False)
// writes it, keys in the order they came.
func StateText(state json.RawMessage) (string, error) {
	var s string
	if err := json.Unmarshal(state, &s); err == nil {
		return s, nil
	}
	v, err := decodeOrdered(state)
	if err != nil {
		return "", err
	}
	return pyDumps(v, false), nil
}

// object is a JSON object with its keys in the order they were written.
type object struct {
	keys   []string
	values []any
}

// decodeOrdered decodes JSON into nil, bool, json.Number, string, []any and
// *object. Empty input is nil.
func decodeOrdered(raw json.RawMessage) (any, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data after the JSON value")
	}
	return v, nil
}

func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			o := &object{}
			for dec.More() {
				k, err := dec.Token()
				if err != nil {
					return nil, err
				}
				v, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				// A repeated key keeps its first place and its last value,
				// as a Python dict does.
				key := k.(string)
				found := false
				for i, have := range o.keys {
					if have == key {
						o.values[i], found = v, true
					}
				}
				if !found {
					o.keys, o.values = append(o.keys, key), append(o.values, v)
				}
			}
			_, err := dec.Token()
			return o, err
		case '[':
			list := []any{}
			for dec.More() {
				v, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				list = append(list, v)
			}
			_, err := dec.Token()
			return list, err
		}
		return nil, fmt.Errorf("unexpected %v", t)
	default:
		return tok, nil
	}
}

// falsy is Python's truth test on a decoded value.
func falsy(v any) bool {
	switch t := v.(type) {
	case nil:
		return true
	case bool:
		return !t
	case string:
		return t == ""
	case json.Number:
		f, err := t.Float64()
		return err == nil && f == 0
	case []any:
		return len(t) == 0
	case *object:
		return len(t.keys) == 0
	}
	return false
}

// pyStr is str(v) for what a criterion may hold.
func pyStr(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case nil:
		return "None"
	case bool:
		if t {
			return "True"
		}
		return "False"
	case json.Number:
		return pyNumber(t)
	}
	// A list or a dict prints with Python's repr, which is not JSON; it is
	// never what a criterion is meant to be, and json.dumps is close enough.
	return pyDumps(v, false)
}

// pyDumps writes a decoded value as json.dumps does with its default
// separators.
func pyDumps(v any, asciiOnly bool) string {
	var b strings.Builder
	var write func(v any)
	write = func(v any) {
		switch t := v.(type) {
		case nil:
			b.WriteString("null")
		case bool:
			b.WriteString(strconv.FormatBool(t))
		case json.Number:
			b.WriteString(pyNumber(t))
		case string:
			pyQuote(&b, t, asciiOnly)
		case []any:
			b.WriteByte('[')
			for i, e := range t {
				if i > 0 {
					b.WriteString(", ")
				}
				write(e)
			}
			b.WriteByte(']')
		case *object:
			b.WriteByte('{')
			for i, k := range t.keys {
				if i > 0 {
					b.WriteString(", ")
				}
				pyQuote(&b, k, asciiOnly)
				b.WriteString(": ")
				write(t.values[i])
			}
			b.WriteByte('}')
		}
	}
	write(v)
	return b.String()
}

// pyQuote is json.dumps's string: a quote, a backslash and the control
// characters escaped, and with asciiOnly everything past ASCII as \uXXXX, in
// surrogate pairs past the first plane.
func pyQuote(b *strings.Builder, s string, asciiOnly bool) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			switch {
			case r < 0x20 || (asciiOnly && r > 0x7e && r < 0x10000):
				fmt.Fprintf(b, `\u%04x`, r)
			case asciiOnly && r >= 0x10000:
				r -= 0x10000
				fmt.Fprintf(b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
			default:
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// pyNumber is how Python writes a JSON number back: an integer as its digits,
// anything with a fraction or an exponent as repr of a float.
func pyNumber(n json.Number) string {
	s := string(n)
	if !strings.ContainsAny(s, ".eE") {
		if i, err := strconv.ParseInt(s, 10, 64); err == nil {
			return strconv.FormatInt(i, 10)
		}
		return strings.TrimPrefix(s, "+")
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return s
	}
	return pyFloat(f)
}

// pyFloat is float.__repr__: the shortest digits that read back, fixed
// between 1e-4 and 1e16 and with at least one decimal, scientific outside with
// a two-digit exponent at least.
func pyFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case math.IsNaN(f):
		return "NaN"
	}
	sci := strconv.FormatFloat(f, 'e', -1, 64)
	mant, expText, _ := strings.Cut(sci, "e")
	exp, _ := strconv.Atoi(expText)
	if exp < -4 || exp >= 16 {
		sign := "+"
		if exp < 0 {
			sign, exp = "-", -exp
		}
		return fmt.Sprintf("%se%s%02d", mant, sign, exp)
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.ContainsAny(s, ".") {
		s += ".0"
	}
	return s
}
