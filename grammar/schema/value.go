package schema

// A JSON value that remembers the order its members were written in.
//
// encoding/json decodes an object into a map, and a map has no order. The order
// of an object's properties decides the order of the rules a schema compiles
// to, and llama.cpp reads its JSON with an ordered map for exactly that reason.
// So this reads the token stream instead and keeps what it saw.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

type kind uint8

const (
	kindNull kind = iota
	kindBool
	kindNumber
	kindString
	kindArray
	kindObject
)

type value struct {
	kind    kind
	boolean bool
	number  json.Number
	text    string
	array   []*value
	members []member
}

type member struct {
	name  string
	value *value
}

// parse reads one JSON document.
func parse(raw []byte) (*value, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := parseValue(dec)
	if err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, fmt.Errorf("more than one JSON document")
	}
	return v, nil
}

func parseValue(dec *json.Decoder) (*value, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	return parseFrom(dec, tok)
}

func parseFrom(dec *json.Decoder, tok json.Token) (*value, error) {
	switch t := tok.(type) {
	case nil:
		return &value{kind: kindNull}, nil
	case bool:
		return &value{kind: kindBool, boolean: t}, nil
	case json.Number:
		return &value{kind: kindNumber, number: t}, nil
	case string:
		return &value{kind: kindString, text: t}, nil
	case json.Delim:
		switch t {
		case '{':
			v := &value{kind: kindObject}
			for dec.More() {
				nameTok, err := dec.Token()
				if err != nil {
					return nil, err
				}
				name, ok := nameTok.(string)
				if !ok {
					return nil, fmt.Errorf("a member name that is not a string")
				}
				member, err := parseValue(dec)
				if err != nil {
					return nil, err
				}
				v.members = append(v.members, memberOf(name, member))
			}
			if _, err := dec.Token(); err != nil { // the closing brace
				return nil, err
			}
			return v, nil
		case '[':
			v := &value{kind: kindArray}
			for dec.More() {
				item, err := parseValue(dec)
				if err != nil {
					return nil, err
				}
				v.array = append(v.array, item)
			}
			if _, err := dec.Token(); err != nil { // the closing bracket
				return nil, err
			}
			return v, nil
		}
	}
	return nil, fmt.Errorf("unexpected %v in the schema", tok)
}

func memberOf(name string, v *value) member { return member{name: name, value: v} }

// get returns a member of an object, or nil.
func (v *value) get(name string) *value {
	if v == nil || v.kind != kindObject {
		return nil
	}
	for _, m := range v.members {
		if m.name == name {
			return m.value
		}
	}
	return nil
}

func (v *value) has(name string) bool { return v.get(name) != nil }

func (v *value) isString() bool { return v != nil && v.kind == kindString }
func (v *value) isObject() bool { return v != nil && v.kind == kindObject }
func (v *value) isArray() bool  { return v != nil && v.kind == kindArray }

// isTrue reports whether the value is the boolean true, which is what an open
// additionalProperties looks like.
func (v *value) isTrue() bool { return v != nil && v.kind == kindBool && v.boolean }

func (v *value) integer(def int) int {
	if v == nil || v.kind != kindNumber {
		return def
	}
	n, err := v.number.Int64()
	if err != nil {
		return def
	}
	return int(n)
}

// dump writes the value back as compact JSON, which is what a constant in a
// schema becomes when it is turned into a literal.
func (v *value) dump() string {
	var b strings.Builder
	v.write(&b)
	return b.String()
}

func (v *value) write(b *strings.Builder) {
	switch v.kind {
	case kindNull:
		b.WriteString("null")
	case kindBool:
		if v.boolean {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case kindNumber:
		b.WriteString(v.number.String())
	case kindString:
		writeJSONString(b, v.text)
	case kindArray:
		b.WriteByte('[')
		for i, item := range v.array {
			if i > 0 {
				b.WriteByte(',')
			}
			item.write(b)
		}
		b.WriteByte(']')
	case kindObject:
		b.WriteByte('{')
		for i, m := range v.members {
			if i > 0 {
				b.WriteByte(',')
			}
			writeJSONString(b, m.name)
			b.WriteByte(':')
			m.value.write(b)
		}
		b.WriteByte('}')
	}
}

// writeJSONString is the escaping a JSON string needs and no more: encoding/json
// would also escape <, > and &, which no other implementation does and which
// would put a different literal in the grammar.
func writeJSONString(b *strings.Builder, s string) {
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
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
				continue
			}
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
}
