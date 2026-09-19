package jinja

// The values a template handles, and how Python would print and compare them.
//
// A template is Python by another name: what it prints is str() of a Python
// object, what it compares is Python equality, and a mapping keeps the order
// its keys went in. So the values here are the Python ones, spelled in Go:
// nil is None, int64 and float64 are the two numbers, []any is a list, Tuple
// a tuple, *Dict a dict that remembers its order, and Undefined what a name
// nobody set stands for.

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Undefined is what a missing name, key or attribute evaluates to. It prints
// as nothing, is false, iterates as empty, and fails as soon as anything is
// asked of it — which is Jinja's default, and what a template's `is defined`
// guards are written against.
type Undefined struct{ hint string }

func (u Undefined) err() error {
	if u.hint == "" {
		return fmt.Errorf("undefined value")
	}
	return fmt.Errorf("%s", u.hint)
}

// Tuple is a list Python would print with parentheses. dictsort and items()
// make them; a template reads them as it reads a list.
type Tuple []any

// Dict is a mapping that keeps its keys in the order they were set, as a
// Python dict does. The keys are strings, numbers, booleans or nil.
type Dict struct {
	keys []any
	vals map[any]any
}

func NewDict() *Dict { return &Dict{vals: map[any]any{}} }

// Set adds a key at the end, or replaces its value where it already is.
func (d *Dict) Set(k, v any) {
	k = normKey(k)
	if _, ok := d.vals[k]; !ok {
		d.keys = append(d.keys, k)
	}
	d.vals[k] = v
}

func (d *Dict) Get(k any) (any, bool) {
	v, ok := d.vals[normKey(k)]
	return v, ok
}

func (d *Dict) Len() int    { return len(d.keys) }
func (d *Dict) Keys() []any { return append([]any(nil), d.keys...) }

// normKey makes 1 and 1.0 the same key, as they are in Python.
func normKey(k any) any {
	if f, ok := k.(float64); ok && f == math.Trunc(f) && math.Abs(f) < 1<<53 {
		return int64(f)
	}
	if b, ok := k.(bool); ok {
		if b {
			return int64(1)
		}
		return int64(0)
	}
	return k
}

// Namespace is what namespace() returns: the one object a template may change
// from inside a loop, because assigning to its attributes crosses scopes.
type Namespace struct{ attrs *Dict }

// truth is Python's bool().
func truth(v any) bool {
	switch v := v.(type) {
	case nil, Undefined:
		return false
	case bool:
		return v
	case int64:
		return v != 0
	case float64:
		return v != 0
	case string:
		return v != ""
	case []any:
		return len(v) > 0
	case Tuple:
		return len(v) > 0
	case *Dict:
		return v.Len() > 0
	}
	return true
}

// str is Python's str(), which is what {{ }} prints.
func str(v any) string {
	switch v := v.(type) {
	case string:
		return v
	case Undefined:
		return ""
	}
	return repr(v)
}

// repr is Python's repr(), which is how a value inside a list is printed.
func repr(v any) string {
	switch v := v.(type) {
	case nil:
		return "None"
	case Undefined:
		return ""
	case bool:
		if v {
			return "True"
		}
		return "False"
	case int64:
		return strconv.FormatInt(v, 10)
	case float64:
		return formatFloat(v)
	case string:
		return reprString(v)
	case []any:
		return "[" + joinRepr(v) + "]"
	case Tuple:
		if len(v) == 1 {
			return "(" + repr(v[0]) + ",)"
		}
		return "(" + joinRepr(v) + ")"
	case *Dict:
		var b strings.Builder
		b.WriteString("{")
		for i, k := range v.keys {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(repr(k) + ": " + repr(v.vals[k]))
		}
		b.WriteString("}")
		return b.String()
	case *Namespace:
		return "<Namespace " + repr(v.attrs) + ">"
	case *loopState:
		return "<LoopContext>"
	}
	return fmt.Sprint(v)
}

func joinRepr(items []any) string {
	parts := make([]string, len(items))
	for i, x := range items {
		parts[i] = repr(x)
	}
	return strings.Join(parts, ", ")
}

// reprString quotes the way Python does: with apostrophes, unless the string
// holds one and no double quote.
func reprString(s string) string {
	q := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		q = '"'
	}
	var b strings.Builder
	b.WriteByte(q)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == rune(q):
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(q)
	return b.String()
}

// formatFloat is Python's repr of a float: the shortest digits that read
// back, in positional notation from 1e-4 up to 1e16, and always with a point
// or an exponent so it cannot be taken for an integer.
func formatFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64) // d.ddde±XX
	mant, exp, _ := strings.Cut(e, "e")
	x, _ := strconv.Atoi(exp)
	if x < -4 || x >= 16 {
		sign := "+"
		if x < 0 {
			sign, x = "-", -x
		}
		return fmt.Sprintf("%se%s%02d", mant, sign, x)
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.ContainsAny(s, ".") {
		s += ".0"
	}
	return s
}

// typeName is how Python names a value's type in an error.
func typeName(v any) string {
	switch v.(type) {
	case nil:
		return "NoneType"
	case Undefined:
		return "Undefined"
	case bool:
		return "bool"
	case int64:
		return "int"
	case float64:
		return "float"
	case string:
		return "str"
	case []any:
		return "list"
	case Tuple:
		return "tuple"
	case *Dict:
		return "dict"
	case *Namespace:
		return "Namespace"
	}
	return fmt.Sprintf("%T", v)
}

// items is what iterating a value yields: a string's characters, a list's
// elements, a mapping's keys.
func items(v any) ([]any, error) {
	switch v := v.(type) {
	case Undefined:
		return nil, nil
	case []any:
		return v, nil
	case Tuple:
		return v, nil
	case *Dict:
		return v.Keys(), nil
	case string:
		out := make([]any, 0, len(v))
		for _, r := range v {
			out = append(out, string(r))
		}
		return out, nil
	}
	return nil, fmt.Errorf("'%s' object is not iterable", typeName(v))
}

// length is Python's len(). A string counts characters, not bytes.
func length(v any) (int, error) {
	switch v := v.(type) {
	case Undefined:
		return 0, nil
	case string:
		return utf8.RuneCountInString(v), nil
	case []any:
		return len(v), nil
	case Tuple:
		return len(v), nil
	case *Dict:
		return v.Len(), nil
	}
	return 0, fmt.Errorf("object of type '%s' has no len()", typeName(v))
}

// equal is Python's ==. 1 == 1.0 == True, and containers compare deeply.
func equal(a, b any) bool {
	if x, ok := number(a); ok {
		if y, ok := number(b); ok {
			return x == y
		}
		return false
	}
	switch a := a.(type) {
	case nil:
		return b == nil
	case Undefined:
		_, ok := b.(Undefined)
		return ok
	case string:
		s, ok := b.(string)
		return ok && a == s
	case []any:
		l, ok := b.([]any)
		return ok && equalSeq(a, l)
	case Tuple:
		l, ok := b.(Tuple)
		return ok && equalSeq(a, l)
	case *Dict:
		d, ok := b.(*Dict)
		if !ok || a.Len() != d.Len() {
			return false
		}
		for _, k := range a.keys {
			w, ok := d.Get(k)
			if !ok || !equal(a.vals[k], w) {
				return false
			}
		}
		return true
	}
	return a == b
}

func equalSeq(a, b []any) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !equal(a[i], b[i]) {
			return false
		}
	}
	return true
}

// number reads the three kinds Python does arithmetic on.
func number(v any) (float64, bool) {
	switch v := v.(type) {
	case int64:
		return float64(v), true
	case float64:
		return v, true
	case bool:
		if v {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// less is Python's <, for the pairs Python orders.
func less(a, b any) (bool, error) {
	if x, ok := number(a); ok {
		if y, ok := number(b); ok {
			if ai, ok := a.(int64); ok {
				if bi, ok := b.(int64); ok {
					return ai < bi, nil
				}
			}
			return x < y, nil
		}
	}
	if x, ok := a.(string); ok {
		if y, ok := b.(string); ok {
			return x < y, nil
		}
	}
	xs, xok := seq(a)
	ys, yok := seq(b)
	if xok && yok {
		for i := 0; i < len(xs) && i < len(ys); i++ {
			if equal(xs[i], ys[i]) {
				continue
			}
			return less(xs[i], ys[i])
		}
		return len(xs) < len(ys), nil
	}
	return false, fmt.Errorf("'<' not supported between instances of '%s' and '%s'", typeName(a), typeName(b))
}

func seq(v any) ([]any, bool) {
	switch v := v.(type) {
	case []any:
		return v, true
	case Tuple:
		return v, true
	}
	return nil, false
}

// sortValues sorts in place with Python's ordering, and reports the first
// pair Python would refuse to compare.
func sortValues(vals []any, key func(any) any) error {
	var err error
	sort.SliceStable(vals, func(i, j int) bool {
		l, e := less(key(vals[i]), key(vals[j]))
		if e != nil && err == nil {
			err = e
		}
		return l
	})
	return err
}
