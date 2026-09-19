package jinja

// Rendering the tree.
//
// Scopes follow Jinja's rules, which are not a reading of the syntax: an if
// opens none, so what it sets is seen after it; each turn of a loop opens its
// own, so nothing it sets survives to the next turn or past the loop; a macro
// runs in a scope of its own above the one it was defined in, whose names it
// reads as they stand when it is called. Namespaces exist because
// of the loop rule, and assigning to one of their attributes is the one write
// that crosses a scope.

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

type scope struct {
	vars   map[string]any
	parent *scope
}

func (s *scope) lookup(name string) (any, bool) {
	for ; s != nil; s = s.parent {
		if v, ok := s.vars[name]; ok {
			return v, true
		}
	}
	return nil, false
}

type renderer struct {
	root *scope
	out  *strings.Builder
}

// The loop controls travel up as errors, and stop at the loop they belong to.
var (
	errBreak    = errors.New("break outside a loop")
	errContinue = errors.New("continue outside a loop")
)

// Error is what raise_exception() raises: the template refusing its input.
type Error struct{ Message string }

func (e *Error) Error() string { return e.Message }

// Render renders the template with the given variables. The globals Jinja and
// transformers define — range, namespace, raise_exception, strftime_now — are
// there unless vars names them itself.
func (t *Template) Render(vars map[string]any) (string, error) {
	globals := &scope{vars: builtinGlobals()}
	root := &scope{vars: map[string]any{}, parent: globals}
	for k, v := range vars {
		root.vars[k] = v
	}
	var b strings.Builder
	r := &renderer{root: root, out: &b}
	if err := r.nodes(t.body, root); err != nil {
		if errors.Is(err, errBreak) || errors.Is(err, errContinue) {
			return "", fmt.Errorf("jinja: %w", err)
		}
		var raised *Error
		if errors.As(err, &raised) {
			return "", err
		}
		return "", fmt.Errorf("jinja: %w", err)
	}
	return b.String(), nil
}

func (r *renderer) nodes(body []node, s *scope) error {
	for _, n := range body {
		if err := r.node(n, s); err != nil {
			return err
		}
	}
	return nil
}

func (r *renderer) node(n node, s *scope) error {
	switch n := n.(type) {
	case textNode:
		r.out.WriteString(n.s)
	case outputNode:
		v, err := r.eval(n.e, s)
		if err != nil {
			return err
		}
		r.out.WriteString(str(v))
	case ifNode:
		for i, c := range n.conds {
			v, err := r.eval(c, s)
			if err != nil {
				return err
			}
			if truth(v) {
				return r.nodes(n.bodies[i], s)
			}
		}
		return r.nodes(n.orElse, s)
	case forNode:
		return r.loop(n, s)
	case setNode:
		v, err := r.eval(n.e, s)
		if err != nil {
			return err
		}
		return r.assign(s, n.targets, n.attr, v)
	case setBlockNode:
		saved := r.out
		var b strings.Builder
		r.out = &b
		err := r.nodes(n.body, s)
		r.out = saved
		if err != nil {
			return err
		}
		return r.assign(s, []string{n.name}, n.attr, b.String())
	case macroNode:
		s.vars[n.name] = &macro{node: n, r: r, def: s}
	case breakNode:
		return errBreak
	case continueNode:
		return errContinue
	default:
		return fmt.Errorf("cannot render %T", n)
	}
	return nil
}

func (r *renderer) assign(s *scope, targets []string, attr string, v any) error {
	if attr != "" {
		obj, ok := s.lookup(targets[0])
		ns, isNS := obj.(*Namespace)
		if !ok || !isNS {
			return fmt.Errorf("cannot assign attribute on non-namespace object")
		}
		ns.attrs.Set(attr, v)
		return nil
	}
	if len(targets) == 1 {
		s.vars[targets[0]] = v
		return nil
	}
	vals, err := items(v)
	if err != nil {
		return err
	}
	if len(vals) != len(targets) {
		return fmt.Errorf("cannot unpack %d values into %d names", len(vals), len(targets))
	}
	for i, t := range targets {
		s.vars[t] = vals[i]
	}
	return nil
}

// loopState is the `loop` a body reads.
type loopState struct {
	index int
	items []any
}

func (l *loopState) attr(name string) (any, bool) {
	n := len(l.items)
	switch name {
	case "index":
		return int64(l.index + 1), true
	case "index0":
		return int64(l.index), true
	case "revindex":
		return int64(n - l.index), true
	case "revindex0":
		return int64(n - l.index - 1), true
	case "first":
		return l.index == 0, true
	case "last":
		return l.index == n-1, true
	case "length":
		return int64(n), true
	case "depth":
		return int64(1), true
	case "depth0":
		return int64(0), true
	case "previtem":
		if l.index > 0 {
			return l.items[l.index-1], true
		}
		return Undefined{hint: "there is no previous item"}, true
	case "nextitem":
		if l.index+1 < n {
			return l.items[l.index+1], true
		}
		return Undefined{hint: "there is no next item"}, true
	case "cycle":
		return builtin(func(args []any, _ *Dict) (any, error) {
			if len(args) == 0 {
				return nil, fmt.Errorf("no items for cycling given")
			}
			return args[l.index%len(args)], nil
		}), true
	}
	return nil, false
}

func (r *renderer) loop(n forNode, s *scope) error {
	iter, err := r.eval(n.iter, s)
	if err != nil {
		return err
	}
	if u, ok := iter.(Undefined); ok && u.hint != "" {
		return u.err()
	}
	all, err := items(iter)
	if err != nil {
		return err
	}
	// The filter runs first, so that loop.length and loop.last count only
	// what passes it.
	var kept []any
	for _, it := range all {
		if n.cond != nil {
			inner := &scope{vars: map[string]any{}, parent: s}
			if err := r.bind(inner, n.targets, it); err != nil {
				return err
			}
			ok, err := r.eval(n.cond, inner)
			if err != nil {
				return err
			}
			if !truth(ok) {
				continue
			}
		}
		kept = append(kept, it)
	}
	if len(kept) == 0 {
		return r.nodes(n.orElse, s)
	}
	state := &loopState{items: kept}
	for i, it := range kept {
		state.index = i
		inner := &scope{vars: map[string]any{"loop": state}, parent: s}
		if err := r.bind(inner, n.targets, it); err != nil {
			return err
		}
		err := r.nodes(n.body, inner)
		if errors.Is(err, errBreak) {
			break
		}
		if err != nil && !errors.Is(err, errContinue) {
			return err
		}
	}
	return nil
}

func (r *renderer) bind(s *scope, targets []string, v any) error {
	if len(targets) == 1 {
		s.vars[targets[0]] = v
		return nil
	}
	return r.assign(s, targets, "", v)
}

// macro is a macro, callable from the template.
type macro struct {
	node macroNode
	r    *renderer
	def  *scope // where it was defined, which it reads as it stands at the call
}

func (m *macro) call(args []any, kwargs *Dict) (any, error) {
	n := m.node
	if len(args) > len(n.params) {
		return nil, fmt.Errorf("macro '%s' takes not more than %d argument(s)", n.name, len(n.params))
	}
	s := &scope{vars: map[string]any{}, parent: m.def}
	for i, p := range n.params {
		switch {
		case i < len(args):
			s.vars[p] = args[i]
		case kwargs != nil && hasKey(kwargs, p):
			v, _ := kwargs.Get(p)
			s.vars[p] = v
		case n.defaults[i] != nil:
			v, err := m.r.eval(n.defaults[i], s)
			if err != nil {
				return nil, err
			}
			s.vars[p] = v
		default:
			s.vars[p] = Undefined{hint: fmt.Sprintf("parameter '%s' was not provided", p)}
		}
	}
	if kwargs != nil {
		for _, k := range kwargs.keys {
			if !contains(n.params, k) {
				return nil, fmt.Errorf("macro '%s' takes no keyword argument '%v'", n.name, k)
			}
		}
	}
	saved := m.r.out
	var b strings.Builder
	m.r.out = &b
	err := m.r.nodes(n.body, s)
	m.r.out = saved
	if err != nil {
		return nil, err
	}
	return b.String(), nil
}

func hasKey(d *Dict, k string) bool {
	_, ok := d.Get(k)
	return ok
}

func contains(list []string, k any) bool {
	for _, x := range list {
		if x == k {
			return true
		}
	}
	return false
}

// builtin is a Go function a template can call.
type builtin func(args []any, kwargs *Dict) (any, error)

func (r *renderer) eval(e expr, s *scope) (any, error) {
	switch e := e.(type) {
	case constExpr:
		return e.v, nil
	case nameExpr:
		if v, ok := s.lookup(e.name); ok {
			return v, nil
		}
		return Undefined{hint: fmt.Sprintf("'%s' is undefined", e.name)}, nil
	case listExpr:
		out := make([]any, len(e.items))
		for i, x := range e.items {
			v, err := r.eval(x, s)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case tupleExpr:
		out := make(Tuple, len(e.items))
		for i, x := range e.items {
			v, err := r.eval(x, s)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case dictExpr:
		d := NewDict()
		for i := range e.keys {
			k, err := r.eval(e.keys[i], s)
			if err != nil {
				return nil, err
			}
			v, err := r.eval(e.vals[i], s)
			if err != nil {
				return nil, err
			}
			d.Set(k, v)
		}
		return d, nil
	case attrExpr:
		obj, err := r.eval(e.obj, s)
		if err != nil {
			return nil, err
		}
		return getAttr(obj, e.name)
	case itemExpr:
		obj, err := r.eval(e.obj, s)
		if err != nil {
			return nil, err
		}
		key, err := r.eval(e.key, s)
		if err != nil {
			return nil, err
		}
		return getItem(obj, key)
	case sliceExpr:
		return r.slice(e, s)
	case callExpr:
		fn, err := r.eval(e.fn, s)
		if err != nil {
			return nil, err
		}
		args, kwargs, err := r.arguments(e.args, e.kwargs, s)
		if err != nil {
			return nil, err
		}
		return call(fn, args, kwargs)
	case filterExpr:
		v, err := r.eval(e.e, s)
		if err != nil {
			return nil, err
		}
		args, kwargs, err := r.arguments(e.args, e.kwargs, s)
		if err != nil {
			return nil, err
		}
		v, err = applyFilter(e.name, v, args, kwargs)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", e.line, err)
		}
		return v, nil
	case testExpr:
		v, err := r.eval(e.e, s)
		if err != nil {
			return nil, err
		}
		args, _, err := r.arguments(e.args, nil, s)
		if err != nil {
			return nil, err
		}
		ok, err := applyTest(e.name, v, args)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", e.line, err)
		}
		return ok != e.negate, nil
	case unaryExpr:
		v, err := r.eval(e.e, s)
		if err != nil {
			return nil, err
		}
		switch v := v.(type) {
		case int64:
			if e.op == "-" {
				return -v, nil
			}
			return v, nil
		case float64:
			if e.op == "-" {
				return -v, nil
			}
			return v, nil
		case bool:
			n := int64(0)
			if v {
				n = 1
			}
			if e.op == "-" {
				return -n, nil
			}
			return n, nil
		}
		return nil, fmt.Errorf("bad operand type for unary %s: '%s'", e.op, typeName(v))
	case binaryExpr:
		a, err := r.eval(e.a, s)
		if err != nil {
			return nil, err
		}
		b, err := r.eval(e.b, s)
		if err != nil {
			return nil, err
		}
		v, err := arith(e.op, a, b)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", e.line, err)
		}
		return v, nil
	case compareExpr:
		a, err := r.eval(e.first, s)
		if err != nil {
			return nil, err
		}
		for i, op := range e.ops {
			b, err := r.eval(e.rest[i], s)
			if err != nil {
				return nil, err
			}
			ok, err := compare(op, a, b)
			if err != nil {
				return nil, fmt.Errorf("line %d: %w", e.line, err)
			}
			if !ok {
				return false, nil
			}
			a = b
		}
		return true, nil
	case andExpr:
		a, err := r.eval(e.a, s)
		if err != nil || !truth(a) {
			return a, err
		}
		return r.eval(e.b, s)
	case orExpr:
		a, err := r.eval(e.a, s)
		if err != nil || truth(a) {
			return a, err
		}
		return r.eval(e.b, s)
	case notExpr:
		v, err := r.eval(e.e, s)
		return !truth(v), err
	case condExpr:
		c, err := r.eval(e.cond, s)
		if err != nil {
			return nil, err
		}
		if truth(c) {
			return r.eval(e.a, s)
		}
		if e.b == nil {
			return Undefined{hint: "the inline if expression evaluated to false and no else section was defined"}, nil
		}
		return r.eval(e.b, s)
	}
	return nil, fmt.Errorf("cannot evaluate %T", e)
}

func (r *renderer) arguments(args []expr, kwargs []kwarg, s *scope) ([]any, *Dict, error) {
	var outArgs []any
	for _, a := range args {
		v, err := r.eval(a, s)
		if err != nil {
			return nil, nil, err
		}
		outArgs = append(outArgs, v)
	}
	var outKw *Dict
	for _, k := range kwargs {
		v, err := r.eval(k.e, s)
		if err != nil {
			return nil, nil, err
		}
		if outKw == nil {
			outKw = NewDict()
		}
		outKw.Set(k.name, v)
	}
	return outArgs, outKw, nil
}

func call(fn any, args []any, kwargs *Dict) (any, error) {
	switch f := fn.(type) {
	case *macro:
		return f.call(args, kwargs)
	case builtin:
		return f(args, kwargs)
	case Undefined:
		return nil, f.err()
	}
	return nil, fmt.Errorf("'%s' object is not callable", typeName(fn))
}

// getAttr is obj.name: an attribute first — a method, a loop's counters, a
// namespace's slots — and a key after that.
func getAttr(obj any, name string) (any, error) {
	switch o := obj.(type) {
	case Undefined:
		return nil, o.err()
	case *Namespace:
		if v, ok := o.attrs.Get(name); ok {
			return v, nil
		}
	case *loopState:
		if v, ok := o.attr(name); ok {
			return v, nil
		}
	case *Dict:
		if m, ok := dictMethod(o, name); ok {
			return m, nil
		}
		if v, ok := o.Get(name); ok {
			return v, nil
		}
		return Undefined{hint: fmt.Sprintf("'dict object' has no attribute '%s'", name)}, nil
	case string:
		if m, ok := stringMethod(o, name); ok {
			return m, nil
		}
	}
	return Undefined{hint: fmt.Sprintf("'%s object' has no attribute '%s'", typeName(obj), name)}, nil
}

// getItem is obj[key]: a key or an index first, and an attribute after that.
func getItem(obj any, key any) (any, error) {
	switch o := obj.(type) {
	case Undefined:
		return nil, o.err()
	case *Dict:
		if v, ok := o.Get(key); ok {
			return v, nil
		}
	case []any, Tuple, string:
		if i, ok := key.(int64); ok {
			list, _ := items(o)
			if i < 0 {
				i += int64(len(list))
			}
			if i >= 0 && i < int64(len(list)) {
				return list[i], nil
			}
			return Undefined{hint: fmt.Sprintf("%s index out of range", typeName(o))}, nil
		}
	}
	if name, ok := key.(string); ok {
		return getAttr(obj, name)
	}
	return Undefined{hint: fmt.Sprintf("'%s object' has no item %s", typeName(obj), repr(key))}, nil
}

func (r *renderer) slice(e sliceExpr, s *scope) (any, error) {
	obj, err := r.eval(e.obj, s)
	if err != nil {
		return nil, err
	}
	var bounds [3]*int64
	for i, x := range []expr{e.lo, e.hi, e.step} {
		if x == nil {
			continue
		}
		v, err := r.eval(x, s)
		if err != nil {
			return nil, err
		}
		if v == nil {
			continue
		}
		n, ok := v.(int64)
		if !ok {
			return nil, fmt.Errorf("slice indices must be integers or None")
		}
		bounds[i] = &n
	}
	if u, ok := obj.(Undefined); ok {
		return nil, u.err()
	}
	list, err := items(obj)
	if err != nil {
		return nil, err
	}
	if _, ok := obj.(*Dict); ok {
		return nil, fmt.Errorf("a dict cannot be sliced")
	}
	picked, err := pySlice(list, bounds[0], bounds[1], bounds[2])
	if err != nil {
		return nil, err
	}
	switch obj.(type) {
	case string:
		var b strings.Builder
		for _, c := range picked {
			b.WriteString(c.(string))
		}
		return b.String(), nil
	case Tuple:
		return Tuple(picked), nil
	}
	return picked, nil
}

// pySlice is Python's list[lo:hi:step].
func pySlice(list []any, lo, hi, step *int64) ([]any, error) {
	n := int64(len(list))
	st := int64(1)
	if step != nil {
		st = *step
	}
	if st == 0 {
		return nil, fmt.Errorf("slice step cannot be zero")
	}
	clamp := func(p *int64, def int64, low, high int64) int64 {
		if p == nil {
			return def
		}
		v := *p
		if v < 0 {
			v += n
		}
		return max(low, min(v, high))
	}
	var out []any
	if st > 0 {
		start, stop := clamp(lo, 0, 0, n), clamp(hi, n, 0, n)
		for i := start; i < stop; i += st {
			out = append(out, list[i])
		}
	} else {
		start, stop := clamp(lo, n-1, -1, n-1), clamp(hi, -1, -1, n-1)
		if hi != nil && *hi < 0 && *hi+n < 0 {
			stop = -1
		}
		for i := start; i > stop; i += st {
			out = append(out, list[i])
		}
	}
	if out == nil {
		out = []any{}
	}
	return out, nil
}

func compare(op string, a, b any) (bool, error) {
	switch op {
	case "==":
		return equal(a, b), nil
	case "!=":
		return !equal(a, b), nil
	case "<":
		return less(a, b)
	case ">":
		return less(b, a)
	case "<=":
		l, err := less(b, a)
		return !l, err
	case ">=":
		l, err := less(a, b)
		return !l, err
	case "in", "not in":
		ok, err := in(a, b)
		if op == "not in" {
			ok = !ok
		}
		return ok, err
	}
	return false, fmt.Errorf("unknown comparison %s", op)
}

// in is Python's `a in b`.
func in(a, b any) (bool, error) {
	switch b := b.(type) {
	case Undefined:
		return false, nil
	case string:
		s, ok := a.(string)
		if !ok {
			return false, fmt.Errorf("'in <string>' requires string as left operand, not %s", typeName(a))
		}
		return strings.Contains(b, s), nil
	case *Dict:
		_, ok := b.Get(a)
		return ok, nil
	case []any, Tuple:
		list, _ := items(b)
		for _, x := range list {
			if equal(a, x) {
				return true, nil
			}
		}
		return false, nil
	}
	return false, fmt.Errorf("argument of type '%s' is not iterable", typeName(b))
}

func arith(op string, a, b any) (any, error) {
	if op == "~" {
		return str(a) + str(b), nil
	}
	for _, v := range []any{a, b} {
		if u, ok := v.(Undefined); ok {
			return nil, u.err()
		}
	}
	ai, aInt := toInt(a)
	bi, bInt := toInt(b)
	af, aNum := number(a)
	bf, bNum := number(b)
	if aNum && bNum {
		both := aInt && bInt
		switch op {
		case "+":
			if both {
				return ai + bi, nil
			}
			return af + bf, nil
		case "-":
			if both {
				return ai - bi, nil
			}
			return af - bf, nil
		case "*":
			if both {
				return ai * bi, nil
			}
			return af * bf, nil
		case "/":
			if bf == 0 {
				return nil, fmt.Errorf("division by zero")
			}
			return af / bf, nil
		case "//":
			if bf == 0 {
				return nil, fmt.Errorf("integer division or modulo by zero")
			}
			if both {
				q := ai / bi
				if (ai%bi != 0) && ((ai < 0) != (bi < 0)) {
					q--
				}
				return q, nil
			}
			return math.Floor(af / bf), nil
		case "%":
			if bf == 0 {
				return nil, fmt.Errorf("integer division or modulo by zero")
			}
			if both {
				m := ai % bi
				if m != 0 && (m < 0) != (bi < 0) {
					m += bi
				}
				return m, nil
			}
			m := math.Mod(af, bf)
			if m != 0 && (m < 0) != (bf < 0) {
				m += bf
			}
			return m, nil
		case "**":
			if both && bi >= 0 {
				out := int64(1)
				for i := int64(0); i < bi; i++ {
					out *= ai
				}
				return out, nil
			}
			return math.Pow(af, bf), nil
		}
	}
	switch op {
	case "+":
		if x, ok := a.(string); ok {
			if y, ok := b.(string); ok {
				return x + y, nil
			}
		}
		if x, ok := seq(a); ok {
			if y, ok := seq(b); ok {
				out := append(append([]any{}, x...), y...)
				if _, isTuple := a.(Tuple); isTuple {
					return Tuple(out), nil
				}
				return out, nil
			}
		}
	case "*":
		if s, ok := a.(string); ok && bInt {
			return strings.Repeat(s, int(max(bi, 0))), nil
		}
		if s, ok := b.(string); ok && aInt {
			return strings.Repeat(s, int(max(ai, 0))), nil
		}
		if x, ok := seq(a); ok && bInt {
			var out []any
			for i := int64(0); i < bi; i++ {
				out = append(out, x...)
			}
			return out, nil
		}
	}
	return nil, fmt.Errorf("unsupported operand type(s) for %s: '%s' and '%s'", op, typeName(a), typeName(b))
}

func toInt(v any) (int64, bool) {
	switch v := v.(type) {
	case int64:
		return v, true
	case bool:
		if v {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}
