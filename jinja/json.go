package jinja

// JSON in and out.
//
// Out, it is what transformers' tojson writes: Python's json.dumps with
// ensure_ascii off, keys in the order they were set, ", " and ": " between
// items, or one item a line when an indent is asked for. Jinja's own tojson
// sorts keys and escapes <, >, & and the apostrophe; transformers replaces it
// precisely so that a template does not print that.
//
// In, ParseJSON keeps the order the keys were written in and reads a number
// without a point or an exponent as an integer, as Python's json.loads does,
// so that a value round-trips through a template unchanged.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

func writeJSON(b *strings.Builder, v any, indent, level int) error {
	newline := func(l int) {
		if indent >= 0 {
			b.WriteString("\n" + strings.Repeat(" ", indent*l))
		}
	}
	sep := ", "
	if indent >= 0 {
		sep = ","
	}
	switch v := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if v {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case int64:
		b.WriteString(strconv.FormatInt(v, 10))
	case float64:
		switch {
		case math.IsNaN(v):
			b.WriteString("NaN")
		case math.IsInf(v, 1):
			b.WriteString("Infinity")
		case math.IsInf(v, -1):
			b.WriteString("-Infinity")
		default:
			b.WriteString(formatFloat(v))
		}
	case string:
		writeJSONString(b, v)
	case []any, Tuple:
		list, _ := seq(v)
		if len(list) == 0 {
			b.WriteString("[]")
			return nil
		}
		b.WriteString("[")
		for i, x := range list {
			if i > 0 {
				b.WriteString(sep)
			}
			newline(level + 1)
			if err := writeJSON(b, x, indent, level+1); err != nil {
				return err
			}
		}
		newline(level)
		b.WriteString("]")
	case *Dict:
		if v.Len() == 0 {
			b.WriteString("{}")
			return nil
		}
		b.WriteString("{")
		for i, k := range v.keys {
			if i > 0 {
				b.WriteString(sep)
			}
			newline(level + 1)
			switch k := k.(type) {
			case string:
				writeJSONString(b, k)
			case nil:
				b.WriteString(`"null"`)
			default:
				var key strings.Builder
				if err := writeJSON(&key, k, -1, 0); err != nil {
					return err
				}
				writeJSONString(b, key.String())
			}
			b.WriteString(": ")
			if err := writeJSON(b, v.vals[k], indent, level+1); err != nil {
				return err
			}
		}
		newline(level)
		b.WriteString("}")
	case *Namespace:
		return fmt.Errorf("Object of type Namespace is not JSON serializable")
	case Undefined:
		return fmt.Errorf("Object of type Undefined is not JSON serializable")
	default:
		return fmt.Errorf("Object of type %s is not JSON serializable", typeName(v))
	}
	return nil
}

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
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// ParseJSON reads one JSON value into template values.
func ParseJSON(data []byte) (any, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	v, err := decodeJSON(d)
	if err != nil {
		return nil, fmt.Errorf("jinja: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, fmt.Errorf("jinja: trailing data after the JSON value")
	}
	return v, nil
}

func decodeJSON(d *json.Decoder) (any, error) {
	t, err := d.Token()
	if err != nil {
		return nil, err
	}
	switch t := t.(type) {
	case json.Delim:
		switch t {
		case '[':
			out := []any{}
			for d.More() {
				v, err := decodeJSON(d)
				if err != nil {
					return nil, err
				}
				out = append(out, v)
			}
			_, err := d.Token()
			return out, err
		case '{':
			out := NewDict()
			for d.More() {
				k, err := d.Token()
				if err != nil {
					return nil, err
				}
				v, err := decodeJSON(d)
				if err != nil {
					return nil, err
				}
				out.Set(k.(string), v)
			}
			_, err := d.Token()
			return out, err
		}
		return nil, fmt.Errorf("unexpected %v", t)
	case json.Number:
		return jsonNumber(string(t))
	}
	return t, nil // string, bool or nil
}

func jsonNumber(s string) (any, error) {
	if !strings.ContainsAny(s, ".eE") {
		if i, err := strconv.ParseInt(s, 10, 64); err == nil {
			return i, nil
		}
	}
	return strconv.ParseFloat(s, 64)
}

// FromGo turns what encoding/json decodes into — maps, slices, float64 — into
// template values. A Go map has no order, so its keys are sorted; a float64
// holding a whole number is an integer, since encoding/json could not tell.
func FromGo(v any) any {
	switch v := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		d := NewDict()
		for _, k := range keys {
			d.Set(k, FromGo(v[k]))
		}
		return d
	case []any:
		out := make([]any, len(v))
		for i, x := range v {
			out[i] = FromGo(x)
		}
		return out
	case []string:
		out := make([]any, len(v))
		for i, x := range v {
			out[i] = x
		}
		return out
	case float64:
		if v == math.Trunc(v) && math.Abs(v) < 1<<53 {
			return int64(v)
		}
		return v
	case json.Number:
		n, err := jsonNumber(string(v))
		if err != nil {
			return string(v)
		}
		return n
	case int:
		return int64(v)
	case int32:
		return int64(v)
	case float32:
		return FromGo(float64(v))
	}
	return v
}
