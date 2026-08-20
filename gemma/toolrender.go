package gemma

// Tools, as the template spells them.
//
// The syntax is the template's own and not JSON: <|"|> is the quote, keys are
// bare, types are upper-cased, and every mapping is walked in sorted key order
// because Jinja's dictsort does. Everything here exists to reproduce that,
// character for character.

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// quoted writes one string the way the template quotes it.
func quoted(text string) string { return quote + text + quote }

// upperQuoted writes a type name, which the template upper-cases.
func upperQuoted(text string) string { return quoted(strings.ToUpper(text)) }

// writeDeclaration writes one tool between <|tool> and <tool|>.
func writeDeclaration(b *strings.Builder, t Tool) error {
	if t.Name == "" {
		return fmt.Errorf("gemma: a tool with no name")
	}
	b.WriteString(toolOpen)
	b.WriteString("declaration:" + t.Name + "{description:" + quoted(t.Description))
	if t.Parameters != nil {
		b.WriteString(",parameters:{")
		writeSchemaBody(b, t.Parameters)
		b.WriteString("}")
	}
	if t.Response != nil {
		b.WriteString(",response:{")
		first := false
		if t.Response.Description != "" {
			b.WriteString("description:" + quoted(t.Response.Description))
			first = true
		}
		if t.Response.Type != "" {
			if first {
				b.WriteString(",")
			}
			b.WriteString("type:" + upperQuoted(t.Response.Type))
		}
		b.WriteString("}")
	}
	b.WriteString("}")
	b.WriteString(toolClose)
	return nil
}

// writeSchemaBody writes properties, required and type, in that order, which is
// the order the template writes them in.
func writeSchemaBody(b *strings.Builder, s *Schema) {
	first := false
	if len(s.Properties) > 0 {
		b.WriteString("properties:{")
		writeProperties(b, s.Properties)
		b.WriteString("}")
		first = true
	}
	if len(s.Required) > 0 {
		if first {
			b.WriteString(",")
		}
		writeRequired(b, s.Required)
		first = true
	}
	if s.Type != "" {
		if first {
			b.WriteString(",")
		}
		b.WriteString("type:" + upperQuoted(s.Type))
	}
}

// writeRequired writes the list of names a schema demands.
func writeRequired(b *strings.Builder, required []string) {
	b.WriteString("required:[")
	for i, name := range required {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(quoted(name))
	}
	b.WriteString("]")
}

// writeProperties writes one entry per property, sorted by name.
func writeProperties(b *strings.Builder, props map[string]*Schema) {
	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	sort.Strings(names)
	for i, name := range names {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(name + ":{")
		writeProperty(b, props[name])
		b.WriteString("}")
	}
}

// writeProperty writes one property's body. type comes last, whatever else is
// there, and each part is separated from the one before it by a comma.
func writeProperty(b *strings.Builder, s *Schema) {
	comma := false
	sep := func() {
		if comma {
			b.WriteString(",")
		}
		comma = true
	}
	if s.Description != "" {
		sep()
		b.WriteString("description:" + quoted(s.Description))
	}
	kind := strings.ToLower(s.Type)
	switch {
	case kind == "string" && len(s.Enum) > 0:
		sep()
		b.WriteString("enum:[")
		for i, v := range s.Enum {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(quoted(v))
		}
		b.WriteString("]")
	case kind == "array" && s.Items != nil:
		sep()
		b.WriteString("items:{")
		writeItems(b, s.Items)
		b.WriteString("}")
	}
	if s.Nullable {
		sep()
		b.WriteString("nullable:true")
	}
	if kind == "object" {
		if len(s.Properties) > 0 {
			sep()
			b.WriteString("properties:{")
			writeProperties(b, s.Properties)
			b.WriteString("}")
		}
		if len(s.Required) > 0 {
			sep()
			writeRequired(b, s.Required)
		}
	}
	sep()
	b.WriteString("type:" + upperQuoted(s.Type))
}

// writeItems writes an array's item schema: the template walks its keys sorted,
// skipping the ones it has no value for.
func writeItems(b *strings.Builder, s *Schema) {
	parts := map[string]string{}
	if s.Description != "" {
		parts["description"] = quoted(s.Description)
	}
	if s.Type != "" {
		parts["type"] = upperQuoted(s.Type)
	}
	keys := make([]string, 0, len(parts))
	for k := range parts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for i, k := range keys {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(k + ":" + parts[k])
	}
}

// writeCall writes one call the model made.
func writeCall(b *strings.Builder, c ToolCall) {
	b.WriteString(toolCallOpen + "call:" + c.Name + "{")
	writeArgumentMap(b, c.Arguments)
	b.WriteString("}" + toolCallClose)
}

// writeResponse writes one tool result. A result that is not a mapping is
// wrapped, as the template wraps it, and a result arriving over HTTP is always
// a string.
func writeResponse(b *strings.Builder, name, content string) {
	b.WriteString(toolResponseOpen + "response:" + name + "{")
	b.WriteString("value:" + quoted(content))
	b.WriteString("}" + toolResponseClose)
}

// writeArgumentMap writes a mapping's entries with bare keys, sorted.
func writeArgumentMap(b *strings.Builder, args map[string]any) {
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for i, k := range keys {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(k + ":")
		writeArgument(b, args[k])
	}
}

// writeArgument writes one value: a string quoted, a boolean as a word, a
// number bare, a list between brackets, a mapping between braces.
func writeArgument(b *strings.Builder, v any) {
	switch v := v.(type) {
	case nil:
		b.WriteString("None")
	case string:
		b.WriteString(quoted(v))
	case bool:
		if v {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case float64:
		b.WriteString(formatNumber(v))
	case int:
		b.WriteString(strconv.Itoa(v))
	case []any:
		b.WriteString("[")
		for i, item := range v {
			if i > 0 {
				b.WriteString(",")
			}
			writeArgument(b, item)
		}
		b.WriteString("]")
	case map[string]any:
		b.WriteString("{")
		writeArgumentMap(b, v)
		b.WriteString("}")
	default:
		fmt.Fprintf(b, "%v", v)
	}
}

// formatNumber writes a number the way the template's Python would: a whole
// float has no fractional part, because JSON gave us a float64 for what was
// written 3.
func formatNumber(f float64) string {
	if f == float64(int64(f)) {
		return strconv.FormatInt(int64(f), 10)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// nameOfCall finds which function a result answers, when the result names only
// the call's identifier.
func nameOfCall(calls []ToolCall, id string) string {
	for _, c := range calls {
		if c.ID == id {
			return c.Name
		}
	}
	return "unknown"
}
