package jinja

// Filters, tests, globals and the methods of strings and mappings.
//
// What is here is what chat templates call, spelled the way Jinja and Python
// spell it, plus the few neighbours a template author reaches for by habit.
// Anything else is refused by name rather than guessed at.

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// param reads one argument, by position or by name.
func param(args []any, kwargs *Dict, i int, name string, def any) any {
	if i < len(args) {
		return args[i]
	}
	if kwargs != nil {
		if v, ok := kwargs.Get(name); ok {
			return v
		}
	}
	return def
}

func builtinGlobals() map[string]any {
	return map[string]any{
		"range": builtin(func(args []any, _ *Dict) (any, error) {
			var lo, hi, step int64 = 0, 0, 1
			ints := make([]int64, len(args))
			for i, a := range args {
				n, ok := toInt(a)
				if !ok {
					return nil, fmt.Errorf("range() takes integers, not '%s'", typeName(a))
				}
				ints[i] = n
			}
			switch len(ints) {
			case 1:
				hi = ints[0]
			case 2:
				lo, hi = ints[0], ints[1]
			case 3:
				lo, hi, step = ints[0], ints[1], ints[2]
			default:
				return nil, fmt.Errorf("range() takes one to three arguments")
			}
			if step == 0 {
				return nil, fmt.Errorf("range() arg 3 must not be zero")
			}
			out := []any{}
			for i := lo; (step > 0 && i < hi) || (step < 0 && i > hi); i += step {
				out = append(out, i)
			}
			return out, nil
		}),
		"namespace": builtin(func(args []any, kwargs *Dict) (any, error) {
			ns := &Namespace{attrs: NewDict()}
			for _, a := range args {
				d, ok := a.(*Dict)
				if !ok {
					return nil, fmt.Errorf("namespace() takes a mapping, not '%s'", typeName(a))
				}
				for _, k := range d.keys {
					ns.attrs.Set(k, d.vals[k])
				}
			}
			if kwargs != nil {
				for _, k := range kwargs.keys {
					ns.attrs.Set(k, kwargs.vals[k])
				}
			}
			return ns, nil
		}),
		"dict": builtin(func(args []any, kwargs *Dict) (any, error) {
			d := NewDict()
			for _, a := range args {
				src, ok := a.(*Dict)
				if !ok {
					return nil, fmt.Errorf("dict() takes a mapping, not '%s'", typeName(a))
				}
				for _, k := range src.keys {
					d.Set(k, src.vals[k])
				}
			}
			if kwargs != nil {
				for _, k := range kwargs.keys {
					d.Set(k, kwargs.vals[k])
				}
			}
			return d, nil
		}),
		"raise_exception": builtin(func(args []any, _ *Dict) (any, error) {
			msg := ""
			if len(args) > 0 {
				msg = str(args[0])
			}
			return nil, &Error{Message: msg}
		}),
		"strftime_now": builtin(func(args []any, _ *Dict) (any, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("strftime_now() takes one format")
			}
			return strftime(time.Now(), str(args[0])), nil
		}),
	}
}

// strftime is the part of C's strftime templates ask for: dates.
func strftime(t time.Time, format string) string {
	var b strings.Builder
	for i := 0; i < len(format); i++ {
		if format[i] != '%' || i+1 == len(format) {
			b.WriteByte(format[i])
			continue
		}
		i++
		switch format[i] {
		case 'Y':
			fmt.Fprintf(&b, "%04d", t.Year())
		case 'y':
			fmt.Fprintf(&b, "%02d", t.Year()%100)
		case 'm':
			fmt.Fprintf(&b, "%02d", int(t.Month()))
		case 'd':
			fmt.Fprintf(&b, "%02d", t.Day())
		case 'H':
			fmt.Fprintf(&b, "%02d", t.Hour())
		case 'M':
			fmt.Fprintf(&b, "%02d", t.Minute())
		case 'S':
			fmt.Fprintf(&b, "%02d", t.Second())
		case 'B':
			b.WriteString(t.Month().String())
		case 'b':
			b.WriteString(t.Month().String()[:3])
		case 'A':
			b.WriteString(t.Weekday().String())
		case 'a':
			b.WriteString(t.Weekday().String()[:3])
		case 'j':
			fmt.Fprintf(&b, "%03d", t.YearDay())
		case '%':
			b.WriteByte('%')
		default:
			b.WriteByte('%')
			b.WriteByte(format[i])
		}
	}
	return b.String()
}

func applyFilter(name string, v any, args []any, kwargs *Dict) (any, error) {
	switch name {
	case "safe", "escape", "e", "forceescape":
		return v, nil
	case "string":
		return str(v), nil
	case "trim":
		chars := param(args, kwargs, 0, "chars", nil)
		if chars == nil {
			return strings.TrimFunc(str(v), pyIsSpace), nil
		}
		return strings.Trim(str(v), str(chars)), nil
	case "upper":
		return strings.ToUpper(str(v)), nil
	case "lower":
		return strings.ToLower(str(v)), nil
	case "capitalize":
		return capitalize(str(v)), nil
	case "title":
		return title(str(v)), nil
	case "length", "count":
		n, err := length(v)
		return int64(n), err
	case "default", "d":
		def := param(args, kwargs, 0, "default_value", "")
		boolean := truth(param(args, kwargs, 1, "boolean", false))
		if _, undefined := v.(Undefined); undefined || (boolean && !truth(v)) {
			return def, nil
		}
		return v, nil
	case "first", "last":
		if u, ok := v.(Undefined); ok {
			return nil, u.err()
		}
		list, err := items(v)
		if err != nil {
			return nil, err
		}
		if len(list) == 0 {
			return Undefined{hint: "No " + name + " item, sequence was empty."}, nil
		}
		if name == "first" {
			return list[0], nil
		}
		return list[len(list)-1], nil
	case "list":
		list, err := items(v)
		return append([]any{}, list...), err
	case "items":
		if _, ok := v.(Undefined); ok {
			return []any{}, nil
		}
		d, ok := v.(*Dict)
		if !ok {
			return nil, fmt.Errorf("can only get item pairs from a mapping")
		}
		return dictItems(d), nil
	case "dictsort":
		d, ok := v.(*Dict)
		if !ok {
			if u, ok := v.(Undefined); ok {
				return nil, u.err()
			}
			return nil, fmt.Errorf("'%s' object has no attribute 'items'", typeName(v))
		}
		caseSensitive := truth(param(args, kwargs, 0, "case_sensitive", false))
		by := str(param(args, kwargs, 1, "by", "key"))
		reverse := truth(param(args, kwargs, 2, "reverse", false))
		pairs := dictItems(d)
		pos := 0
		if by == "value" {
			pos = 1
		}
		err := sortValues(pairs, func(x any) any {
			k := x.(Tuple)[pos]
			if s, ok := k.(string); ok && !caseSensitive {
				return strings.ToLower(s)
			}
			return k
		})
		if reverse {
			reverseList(pairs)
		}
		return pairs, err
	case "join":
		list, err := items(v)
		if err != nil {
			return nil, err
		}
		sep := str(param(args, kwargs, 0, "d", ""))
		attr := param(args, kwargs, 1, "attribute", nil)
		parts := make([]string, len(list))
		for i, x := range list {
			if attr != nil {
				if x, err = getItem(x, attr); err != nil {
					return nil, err
				}
			}
			parts[i] = str(x)
		}
		return strings.Join(parts, sep), nil
	case "reverse":
		if s, ok := v.(string); ok {
			r := []rune(s)
			for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
				r[i], r[j] = r[j], r[i]
			}
			return string(r), nil
		}
		list, err := items(v)
		if err != nil {
			return nil, err
		}
		out := append([]any{}, list...)
		reverseList(out)
		return out, nil
	case "sort":
		list, err := items(v)
		if err != nil {
			return nil, err
		}
		out := append([]any{}, list...)
		reverse := truth(param(args, kwargs, 0, "reverse", false))
		caseSensitive := truth(param(args, kwargs, 1, "case_sensitive", false))
		attr := param(args, kwargs, 2, "attribute", nil)
		var keyErr error
		err = sortValues(out, func(x any) any {
			if attr != nil {
				var e error
				if x, e = getItem(x, attr); e != nil && keyErr == nil {
					keyErr = e
				}
			}
			if s, ok := x.(string); ok && !caseSensitive {
				return strings.ToLower(s)
			}
			return x
		})
		if reverse {
			reverseList(out)
		}
		if keyErr != nil {
			return nil, keyErr
		}
		return out, err
	case "unique":
		list, err := items(v)
		if err != nil {
			return nil, err
		}
		var out []any
		for _, x := range list {
			seen := false
			for _, y := range out {
				if equal(x, y) {
					seen = true
					break
				}
			}
			if !seen {
				out = append(out, x)
			}
		}
		return out, nil
	case "map":
		return mapFilter(v, args, kwargs)
	case "select", "reject", "selectattr", "rejectattr":
		return selectFilter(name, v, args)
	case "replace":
		old, repl := str(param(args, kwargs, 0, "old", "")), str(param(args, kwargs, 1, "new", ""))
		count, _ := toInt(param(args, kwargs, 2, "count", int64(-1)))
		return strings.Replace(str(v), old, repl, int(count)), nil
	case "abs":
		switch n := v.(type) {
		case int64:
			if n < 0 {
				return -n, nil
			}
			return n, nil
		case float64:
			return math.Abs(n), nil
		}
		return nil, fmt.Errorf("bad operand type for abs(): '%s'", typeName(v))
	case "int":
		def := param(args, kwargs, 0, "default", int64(0))
		switch n := v.(type) {
		case int64:
			return n, nil
		case bool:
			i, _ := toInt(n)
			return i, nil
		case float64:
			return int64(n), nil
		case string:
			if i, err := strconv.ParseInt(strings.TrimSpace(n), 10, 64); err == nil {
				return i, nil
			}
			if f, err := strconv.ParseFloat(strings.TrimSpace(n), 64); err == nil {
				return int64(f), nil
			}
		}
		return def, nil
	case "float":
		def := param(args, kwargs, 0, "default", 0.0)
		if f, ok := number(v); ok {
			return f, nil
		}
		if s, ok := v.(string); ok {
			if f, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil {
				return f, nil
			}
		}
		return def, nil
	case "round":
		f, ok := number(v)
		if !ok {
			return nil, fmt.Errorf("round() takes a number")
		}
		prec, _ := toInt(param(args, kwargs, 0, "precision", int64(0)))
		method := str(param(args, kwargs, 1, "method", "common"))
		p := math.Pow(10, float64(prec))
		switch method {
		case "ceil":
			return math.Ceil(f*p) / p, nil
		case "floor":
			return math.Floor(f*p) / p, nil
		}
		if prec >= 0 {
			// Python rounds the exact binary value, so 2.675, which is stored
			// a hair under, goes to 2.67; FormatFloat rounds the same way.
			r, _ := strconv.ParseFloat(strconv.FormatFloat(f, 'f', int(prec), 64), 64)
			return r, nil
		}
		return math.RoundToEven(f*p) / p, nil
	case "sum":
		list, err := items(v)
		if err != nil {
			return nil, err
		}
		attr := param(args, kwargs, 0, "attribute", nil)
		var total any = param(args, kwargs, 1, "start", int64(0))
		for _, x := range list {
			if attr != nil {
				if x, err = getItem(x, attr); err != nil {
					return nil, err
				}
			}
			if total, err = arith("+", total, x); err != nil {
				return nil, err
			}
		}
		return total, nil
	case "min", "max":
		list, err := items(v)
		if err != nil {
			return nil, err
		}
		if len(list) == 0 {
			return Undefined{hint: "No " + name + " item, sequence was empty."}, nil
		}
		best := list[0]
		for _, x := range list[1:] {
			l, err := less(x, best)
			if err != nil {
				return nil, err
			}
			if g, _ := less(best, x); (name == "min" && l) || (name == "max" && g) {
				best = x
			}
		}
		return best, nil
	case "tojson":
		indent := param(args, kwargs, 0, "indent", nil)
		n := -1
		if indent != nil {
			i, ok := toInt(indent)
			if !ok {
				return nil, fmt.Errorf("tojson's indent must be an integer")
			}
			n = int(i)
		}
		var b strings.Builder
		if err := writeJSON(&b, v, n, 0); err != nil {
			return nil, err
		}
		return b.String(), nil
	case "indent":
		width := param(args, kwargs, 0, "width", int64(4))
		first := truth(param(args, kwargs, 1, "first", false))
		blank := truth(param(args, kwargs, 2, "blank", false))
		pad := ""
		if s, ok := width.(string); ok {
			pad = s
		} else {
			w, _ := toInt(width)
			pad = strings.Repeat(" ", int(w))
		}
		lines := strings.Split(str(v), "\n")
		for i := range lines {
			if (i == 0 && !first) || (!blank && strings.TrimFunc(lines[i], pyIsSpace) == "") {
				continue
			}
			lines[i] = pad + lines[i]
		}
		return strings.Join(lines, "\n"), nil
	case "center":
		w, _ := toInt(param(args, kwargs, 0, "width", int64(80)))
		s := str(v)
		n := int(w) - utf8.RuneCountInString(s)
		if n <= 0 {
			return s, nil
		}
		left := n / 2
		if n%2 == 1 && int(w)%2 == 1 {
			left++
		}
		return strings.Repeat(" ", left) + s + strings.Repeat(" ", n-left), nil
	case "wordcount":
		return int64(len(strings.FieldsFunc(str(v), pyIsSpace))), nil
	case "attr":
		return getAttr(v, str(param(args, kwargs, 0, "name", "")))
	case "batch", "slice", "groupby", "filesizeformat", "format", "pprint", "striptags", "truncate", "urlencode", "urlize", "wordwrap", "xmlattr":
		return nil, fmt.Errorf("the filter %q is not implemented", name)
	}
	return nil, fmt.Errorf("no filter named '%s'", name)
}

func dictItems(d *Dict) []any {
	out := make([]any, len(d.keys))
	for i, k := range d.keys {
		out[i] = Tuple{k, d.vals[k]}
	}
	return out
}

func reverseList(l []any) {
	for i, j := 0, len(l)-1; i < j; i, j = i+1, j-1 {
		l[i], l[j] = l[j], l[i]
	}
}

// mapFilter is `map(attribute='x')` or `map('filter', args...)`.
func mapFilter(v any, args []any, kwargs *Dict) (any, error) {
	list, err := items(v)
	if err != nil {
		return nil, err
	}
	out := make([]any, 0, len(list))
	if attr, ok := kwargsGet(kwargs, "attribute"); ok {
		def, hasDef := kwargsGet(kwargs, "default")
		for _, x := range list {
			y, err := getItem(x, attr)
			if err != nil {
				return nil, err
			}
			if _, undefined := y.(Undefined); undefined && hasDef {
				y = def
			}
			out = append(out, y)
		}
		return out, nil
	}
	if len(args) == 0 {
		return nil, fmt.Errorf("map() needs a filter name or an attribute")
	}
	name := str(args[0])
	for _, x := range list {
		y, err := applyFilter(name, x, args[1:], kwargs)
		if err != nil {
			return nil, err
		}
		out = append(out, y)
	}
	return out, nil
}

func kwargsGet(kwargs *Dict, name string) (any, bool) {
	if kwargs == nil {
		return nil, false
	}
	return kwargs.Get(name)
}

// selectFilter is select, reject, selectattr and rejectattr: keep what passes
// a test, or what fails it, of an item or of one of its attributes.
func selectFilter(name string, v any, args []any) (any, error) {
	list, err := items(v)
	if err != nil {
		return nil, err
	}
	byAttr := strings.HasSuffix(name, "attr")
	keep := strings.HasPrefix(name, "select")
	var attr any
	if byAttr {
		if len(args) == 0 {
			return nil, fmt.Errorf("%s() needs an attribute", name)
		}
		attr, args = args[0], args[1:]
	}
	var out []any
	for _, x := range list {
		subject := x
		if byAttr {
			if subject, err = getItem(x, attr); err != nil {
				return nil, err
			}
		}
		var ok bool
		if len(args) == 0 {
			ok = truth(subject)
		} else if ok, err = applyTest(str(args[0]), subject, args[1:]); err != nil {
			return nil, err
		}
		if ok == keep {
			out = append(out, x)
		}
	}
	if out == nil {
		out = []any{}
	}
	return out, nil
}

func applyTest(name string, v any, args []any) (bool, error) {
	arg := func() (any, error) {
		if len(args) == 0 {
			return nil, fmt.Errorf("the test %s takes an argument", name)
		}
		return args[0], nil
	}
	switch name {
	case "defined":
		_, u := v.(Undefined)
		return !u, nil
	case "undefined":
		_, u := v.(Undefined)
		return u, nil
	case "none":
		return v == nil, nil
	case "string":
		_, ok := v.(string)
		return ok, nil
	case "mapping":
		_, ok := v.(*Dict)
		return ok, nil
	case "iterable":
		switch v.(type) {
		case string, []any, Tuple, *Dict:
			return true, nil
		}
		return false, nil
	case "sequence":
		switch v.(type) {
		case string, []any, Tuple, *Dict:
			return true, nil
		}
		return false, nil
	case "number":
		switch v.(type) {
		case int64, float64, bool:
			return true, nil
		}
		return false, nil
	case "integer":
		_, ok := v.(int64)
		return ok, nil
	case "float":
		_, ok := v.(float64)
		return ok, nil
	case "boolean":
		_, ok := v.(bool)
		return ok, nil
	case "true":
		b, ok := v.(bool)
		return ok && b, nil
	case "false":
		b, ok := v.(bool)
		return ok && !b, nil
	case "callable":
		switch v.(type) {
		case *macro, builtin:
			return true, nil
		}
		return false, nil
	case "lower":
		s := str(v)
		return s == strings.ToLower(s), nil
	case "upper":
		s := str(v)
		return s == strings.ToUpper(s), nil
	case "even", "odd":
		n, ok := toInt(v)
		if !ok {
			return false, fmt.Errorf("the test %s takes an integer", name)
		}
		return (n%2 == 0) == (name == "even"), nil
	case "divisibleby":
		a, err := arg()
		if err != nil {
			return false, err
		}
		m, err := arith("%", v, a)
		if err != nil {
			return false, err
		}
		return equal(m, int64(0)), nil
	case "eq", "equalto", "==":
		a, err := arg()
		return err == nil && equal(v, a), err
	case "ne", "!=":
		a, err := arg()
		return err == nil && !equal(v, a), err
	case "lt", "lessthan", "<", "gt", "greaterthan", ">", "le", "<=", "ge", ">=":
		a, err := arg()
		if err != nil {
			return false, err
		}
		op := map[string]string{"lt": "<", "lessthan": "<", "gt": ">", "greaterthan": ">", "le": "<=", "ge": ">="}[name]
		if op == "" {
			op = name
		}
		return compare(op, v, a)
	case "in":
		a, err := arg()
		if err != nil {
			return false, err
		}
		return in(v, a)
	case "sameas":
		a, err := arg()
		if err != nil {
			return false, err
		}
		switch v.(type) {
		case nil, bool:
			return v == a, nil
		}
		return equal(v, a), nil
	}
	return false, fmt.Errorf("no test named '%s'", name)
}

func dictMethod(d *Dict, name string) (any, bool) {
	switch name {
	case "get":
		return builtin(func(args []any, kwargs *Dict) (any, error) {
			if len(args) == 0 {
				return nil, fmt.Errorf("get expected at least 1 argument")
			}
			if v, ok := d.Get(args[0]); ok {
				return v, nil
			}
			return param(args, kwargs, 1, "default", nil), nil
		}), true
	case "items":
		return builtin(func([]any, *Dict) (any, error) { return dictItems(d), nil }), true
	case "keys":
		return builtin(func([]any, *Dict) (any, error) { return d.Keys(), nil }), true
	case "values":
		return builtin(func([]any, *Dict) (any, error) {
			out := make([]any, len(d.keys))
			for i, k := range d.keys {
				out[i] = d.vals[k]
			}
			return out, nil
		}), true
	}
	return nil, false
}

func stringMethod(s string, name string) (any, bool) {
	strArg := func(args []any, i int) (string, bool) {
		if i >= len(args) || args[i] == nil {
			return "", false
		}
		return str(args[i]), true
	}
	switch name {
	case "startswith", "endswith":
		return builtin(func(args []any, _ *Dict) (any, error) {
			if len(args) == 0 {
				return nil, fmt.Errorf("%s takes at least 1 argument", name)
			}
			var prefixes []any
			if t, ok := seq(args[0]); ok {
				prefixes = t
			} else {
				prefixes = []any{args[0]}
			}
			for _, p := range prefixes {
				ps, ok := p.(string)
				if !ok {
					return nil, fmt.Errorf("%s first arg must be str or a tuple of str", name)
				}
				if (name == "startswith" && strings.HasPrefix(s, ps)) || (name == "endswith" && strings.HasSuffix(s, ps)) {
					return true, nil
				}
			}
			return false, nil
		}), true
	case "strip", "lstrip", "rstrip":
		return builtin(func(args []any, _ *Dict) (any, error) {
			chars, has := strArg(args, 0)
			trim := func(t string) string {
				if !has {
					switch name {
					case "lstrip":
						return strings.TrimLeftFunc(t, pyIsSpace)
					case "rstrip":
						return strings.TrimRightFunc(t, pyIsSpace)
					}
					return strings.TrimFunc(t, pyIsSpace)
				}
				switch name {
				case "lstrip":
					return strings.TrimLeft(t, chars)
				case "rstrip":
					return strings.TrimRight(t, chars)
				}
				return strings.Trim(t, chars)
			}
			return trim(s), nil
		}), true
	case "split", "rsplit":
		return builtin(func(args []any, kwargs *Dict) (any, error) {
			sep := param(args, kwargs, 0, "sep", nil)
			limit, _ := toInt(param(args, kwargs, 1, "maxsplit", int64(-1)))
			return pySplit(s, sep, int(limit), name == "rsplit")
		}), true
	case "splitlines":
		return builtin(func([]any, *Dict) (any, error) {
			lines := strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n")
			if len(lines) > 0 && lines[len(lines)-1] == "" {
				lines = lines[:len(lines)-1]
			}
			out := make([]any, len(lines))
			for i, l := range lines {
				out[i] = l
			}
			return out, nil
		}), true
	case "replace":
		return builtin(func(args []any, _ *Dict) (any, error) {
			if len(args) < 2 {
				return nil, fmt.Errorf("replace takes at least 2 arguments")
			}
			n := int64(-1)
			if len(args) > 2 {
				n, _ = toInt(args[2])
			}
			return strings.Replace(s, str(args[0]), str(args[1]), int(n)), nil
		}), true
	case "upper":
		return builtin(func([]any, *Dict) (any, error) { return strings.ToUpper(s), nil }), true
	case "lower":
		return builtin(func([]any, *Dict) (any, error) { return strings.ToLower(s), nil }), true
	case "title":
		return builtin(func([]any, *Dict) (any, error) { return title(s), nil }), true
	case "capitalize":
		return builtin(func([]any, *Dict) (any, error) { return capitalize(s), nil }), true
	case "find", "rfind", "index", "rindex", "count":
		return builtin(func(args []any, _ *Dict) (any, error) {
			sub, ok := strArg(args, 0)
			if !ok {
				return nil, fmt.Errorf("%s takes a string", name)
			}
			if name == "count" {
				if sub == "" {
					return int64(utf8.RuneCountInString(s) + 1), nil
				}
				return int64(strings.Count(s, sub)), nil
			}
			at := strings.Index(s, sub)
			if name[0] == 'r' {
				at = strings.LastIndex(s, sub)
			}
			if at < 0 {
				if name == "index" || name == "rindex" {
					return nil, fmt.Errorf("substring not found")
				}
				return int64(-1), nil
			}
			return int64(utf8.RuneCountInString(s[:at])), nil
		}), true
	case "isdigit", "isalpha", "isalnum", "isspace", "isupper", "islower":
		return builtin(func([]any, *Dict) (any, error) {
			if s == "" {
				return false, nil
			}
			switch name {
			case "isupper":
				return s == strings.ToUpper(s) && s != strings.ToLower(s), nil
			case "islower":
				return s == strings.ToLower(s) && s != strings.ToUpper(s), nil
			}
			for _, r := range s {
				ok := map[string]bool{
					"isdigit": unicode.IsDigit(r),
					"isalpha": unicode.IsLetter(r),
					"isalnum": unicode.IsLetter(r) || unicode.IsDigit(r),
					"isspace": pyIsSpace(r),
				}[name]
				if !ok {
					return false, nil
				}
			}
			return true, nil
		}), true
	case "join":
		return builtin(func(args []any, _ *Dict) (any, error) {
			if len(args) != 1 {
				return nil, fmt.Errorf("join takes one argument")
			}
			list, err := items(args[0])
			if err != nil {
				return nil, err
			}
			parts := make([]string, len(list))
			for i, x := range list {
				p, ok := x.(string)
				if !ok {
					return nil, fmt.Errorf("sequence item %d: expected str instance, %s found", i, typeName(x))
				}
				parts[i] = p
			}
			return strings.Join(parts, s), nil
		}), true
	}
	return nil, false
}

// pySplit is str.split and str.rsplit. With no separator, runs of blanks
// separate and blanks at either end make no empty field.
func pySplit(s string, sep any, limit int, fromRight bool) (any, error) {
	var parts []string
	if sep == nil {
		// With a limit, what is left after the last cut keeps its blanks on
		// the far side.
		switch {
		case limit < 0:
			parts = strings.FieldsFunc(s, pyIsSpace)
		case !fromRight:
			rest := strings.TrimLeftFunc(s, pyIsSpace)
			for i := 0; i < limit && rest != ""; i++ {
				end := strings.IndexFunc(rest, pyIsSpace)
				if end < 0 {
					break
				}
				parts = append(parts, rest[:end])
				rest = strings.TrimLeftFunc(rest[end:], pyIsSpace)
			}
			if rest != "" {
				parts = append(parts, rest)
			}
		default:
			rest := strings.TrimRightFunc(s, pyIsSpace)
			var tail []string
			for i := 0; i < limit && rest != ""; i++ {
				at := strings.LastIndexFunc(rest, pyIsSpace)
				if at < 0 {
					break
				}
				_, size := utf8.DecodeRuneInString(rest[at:])
				tail = append([]string{rest[at+size:]}, tail...)
				rest = strings.TrimRightFunc(rest[:at], pyIsSpace)
			}
			if rest != "" {
				tail = append([]string{rest}, tail...)
			}
			parts = tail
		}
	} else {
		sp := str(sep)
		if sp == "" {
			return nil, fmt.Errorf("empty separator")
		}
		switch {
		case limit < 0:
			parts = strings.Split(s, sp)
		case !fromRight:
			parts = strings.SplitN(s, sp, limit+1)
		default:
			all := strings.Split(s, sp)
			if len(all) <= limit+1 {
				parts = all
			} else {
				cut := len(all) - limit
				parts = append([]string{strings.Join(all[:cut], sp)}, all[cut:]...)
			}
		}
	}
	out := make([]any, len(parts))
	for i, p := range parts {
		out[i] = p
	}
	return out, nil
}

// pyIsSpace is Python's str.isspace, which counts the four separator controls
// Go's unicode.IsSpace does not.
func pyIsSpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	r, n := utf8.DecodeRuneInString(s)
	return string(unicode.ToUpper(r)) + strings.ToLower(s[n:])
}

// title upper-cases the first letter of every word and lower-cases the rest,
// a word being a run of letters, as Python's str.title has it.
func title(s string) string {
	var b strings.Builder
	prev := false
	for _, r := range s {
		if unicode.IsLetter(r) {
			if prev {
				b.WriteRune(unicode.ToLower(r))
			} else {
				b.WriteRune(unicode.ToUpper(r))
			}
			prev = true
			continue
		}
		prev = false
		b.WriteRune(r)
	}
	return b.String()
}
