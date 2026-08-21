package qwen

// Tools, as the template spells them.
//
// The template writes a declaration with Jinja's tojson, and encoding/json is
// not that: tojson sorts an object's keys, separates with ", " and ": ", and
// escapes the four characters HTML cares about — including the apostrophe,
// which JSON does not require and Go does not write. So the JSON here is
// emitted by hand, character for character, the way gemma/toolrender.go emits
// that template's own syntax.

import (
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/ThiraSoft/golem/chat"
)

// writeDeclaration writes one function the way the template declares it.
//
// A tool whose parameters are nil has no "parameters" key at all, which is what
// a function taking none looks like on the wire. An empty description is
// written rather than dropped: chat.Tool cannot tell an absent description from
// an empty one, and every real declaration carries one.
func writeDeclaration(b *strings.Builder, t chat.Tool) {
	function := map[string]any{
		"name":        t.Name,
		"description": t.Description,
	}
	if t.Parameters != nil {
		function["parameters"] = schemaValue(t.Parameters)
	}
	if t.Response != nil {
		function["response"] = schemaValue(t.Response)
	}
	writeJSON(b, map[string]any{"type": "function", "function": function})
}

// writeCall writes one call the way the model wrote it, so a conversation fed
// back reads the same as the one that produced it.
func writeCall(b *strings.Builder, call chat.ToolCall) {
	b.WriteString(toolCallOpen + "\n{\"name\": ")
	writeJSON(b, call.Name)
	b.WriteString(", \"arguments\": ")
	args := call.Arguments
	if args == nil {
		args = map[string]any{}
	}
	writeJSON(b, args)
	b.WriteString("}\n" + toolCallClose)
}

// schemaValue turns a schema into the mapping the wire spells it with, keeping
// the omitempty rules chat.Schema declares.
func schemaValue(s *chat.Schema) map[string]any {
	out := map[string]any{}
	if s.Type != "" {
		out["type"] = s.Type
	}
	if s.Description != "" {
		out["description"] = s.Description
	}
	if len(s.Properties) > 0 {
		properties := map[string]any{}
		for name, property := range s.Properties {
			properties[name] = schemaValue(property)
		}
		out["properties"] = properties
	}
	if len(s.Required) > 0 {
		out["required"] = anyStrings(s.Required)
	}
	if s.Items != nil {
		out["items"] = schemaValue(s.Items)
	}
	if len(s.Enum) > 0 {
		out["enum"] = anyStrings(s.Enum)
	}
	if s.Nullable {
		out["nullable"] = true
	}
	return out
}

// anyStrings widens a list of strings, because writeJSON walks []any.
func anyStrings(values []string) []any {
	out := make([]any, len(values))
	for i, v := range values {
		out[i] = v
	}
	return out
}

// writeJSON writes one value the way Jinja's tojson does.
func writeJSON(b *strings.Builder, v any) {
	switch v := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if v {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case string:
		writeJSONString(b, v)
	case float64:
		// A number that came back from JSON is a float64 here and was an int
		// in Python, which writes it without a fractional part.
		if v == math.Trunc(v) && math.Abs(v) < 1<<53 {
			b.WriteString(strconv.FormatInt(int64(v), 10))
		} else {
			b.WriteString(strconv.FormatFloat(v, 'g', -1, 64))
		}
	case int:
		b.WriteString(strconv.Itoa(v))
	case []any:
		b.WriteString("[")
		for i, item := range v {
			if i > 0 {
				b.WriteString(", ")
			}
			writeJSON(b, item)
		}
		b.WriteString("]")
	case map[string]any:
		keys := make([]string, 0, len(v))
		for key := range v {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		b.WriteString("{")
		for i, key := range keys {
			if i > 0 {
				b.WriteString(", ")
			}
			writeJSONString(b, key)
			b.WriteString(": ")
			writeJSON(b, v[key])
		}
		b.WriteString("}")
	default:
		// Nothing else reaches here: the values come from chat.Tool and from
		// arguments that encoding/json decoded, and those are the cases above.
		b.WriteString("null")
	}
}

// writeJSONString quotes a string as json.dumps does with ensure_ascii, then as
// Jinja escapes it: anything outside ASCII becomes \uXXXX, and the four
// characters HTML cares about become escapes too.
func writeJSONString(b *strings.Builder, text string) {
	b.WriteByte('"')
	for _, r := range text {
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
		case '<':
			b.WriteString(`\u003c`)
		case '>':
			b.WriteString(`\u003e`)
		case '&':
			b.WriteString(`\u0026`)
		case '\'':
			b.WriteString(`\u0027`)
		default:
			switch {
			case r < 0x20:
				writeEscape(b, uint16(r))
			case r < 0x7f:
				b.WriteRune(r)
			case r > 0xffff:
				// Outside the basic plane, json.dumps writes a surrogate pair.
				r -= 0x10000
				writeEscape(b, uint16(0xd800+(r>>10)))
				writeEscape(b, uint16(0xdc00+(r&0x3ff)))
			default:
				writeEscape(b, uint16(r))
			}
		}
	}
	b.WriteByte('"')
}

func writeEscape(b *strings.Builder, code uint16) {
	const hex = "0123456789abcdef"
	b.WriteString(`\u`)
	b.WriteByte(hex[code>>12])
	b.WriteByte(hex[(code>>8)&0xf])
	b.WriteByte(hex[(code>>4)&0xf])
	b.WriteByte(hex[code&0xf])
}
