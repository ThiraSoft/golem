package jinja

// From tokens to a tree.
//
// The grammar is Jinja's, in Jinja's order of precedence, from loosest to
// tightest: a conditional expression, or, and, not, comparisons, + and -, ~,
// *, /, // and %, **, a sign, and last what follows a primary — attributes,
// subscripts, calls, filters and tests, left to right. That last level is
// why `x | length > 0` compares a length, and why `-1 | abs` is 1.

import "fmt"

type node interface{}

type (
	textNode   struct{ s string }
	outputNode struct{ e expr }
	ifNode     struct {
		conds  []expr
		bodies [][]node
		orElse []node
	}
	forNode struct {
		targets []string
		iter    expr
		cond    expr // the `if` after the iterable, or nil
		body    []node
		orElse  []node
	}
	// setNode assigns to names, or to one attribute of a namespace.
	setNode struct {
		targets []string
		attr    string
		e       expr
	}
	setBlockNode struct {
		name, attr string
		body       []node
	}
	macroNode struct {
		name     string
		params   []string
		defaults []expr // nil where a parameter has none
		body     []node
	}
	breakNode    struct{}
	continueNode struct{}
)

type expr interface{}

type (
	constExpr struct{ v any }
	nameExpr  struct {
		name string
		line int
	}
	listExpr  struct{ items []expr }
	tupleExpr struct{ items []expr }
	dictExpr  struct{ keys, vals []expr }
	attrExpr  struct {
		obj  expr
		name string
	}
	itemExpr  struct{ obj, key expr }
	sliceExpr struct {
		obj          expr
		lo, hi, step expr
	}
	callExpr struct {
		fn     expr
		args   []expr
		kwargs []kwarg
	}
	filterExpr struct {
		e      expr
		name   string
		args   []expr
		kwargs []kwarg
		line   int
	}
	testExpr struct {
		e      expr
		name   string
		args   []expr
		negate bool
		line   int
	}
	unaryExpr struct {
		op string
		e  expr
	}
	binaryExpr struct {
		op   string
		a, b expr
		line int
	}
	compareExpr struct {
		first expr
		ops   []string
		rest  []expr
		line  int
	}
	andExpr  struct{ a, b expr }
	orExpr   struct{ a, b expr }
	notExpr  struct{ e expr }
	condExpr struct{ cond, a, b expr } // b nil when there is no else
)

type kwarg struct {
	name string
	e    expr
}

// Template is a parsed template, ready to render as many times as asked.
type Template struct {
	body []node
}

// Parse reads a template. Everything is checked here that can be without the
// values it will be rendered with.
func Parse(src string) (*Template, error) {
	chunks, err := lex(src)
	if err != nil {
		return nil, fmt.Errorf("jinja: %w", err)
	}
	p := &parser{chunks: chunks}
	body, end, err := p.statements()
	if err != nil {
		return nil, fmt.Errorf("jinja: %w", err)
	}
	if end != "" {
		return nil, fmt.Errorf("jinja: line %d: %q closes nothing", p.line, end)
	}
	return &Template{body: body}, nil
}

type parser struct {
	chunks []chunk
	at     int
	line   int
	// Within one tag.
	toks []token
	pos  int
}

// statements reads until a block tag it does not own — an end, an else — and
// returns that tag's name with the tokens positioned after it.
func (p *parser) statements() ([]node, string, error) {
	var out []node
	for p.at < len(p.chunks) {
		c := p.chunks[p.at]
		p.at++
		p.line = c.line
		switch c.kind {
		case chunkText:
			out = append(out, textNode{c.text})
		case chunkVar:
			p.toks, p.pos = c.toks, 0
			e, err := p.expression()
			if err != nil {
				return nil, "", err
			}
			if err := p.end(); err != nil {
				return nil, "", err
			}
			out = append(out, outputNode{e})
		case chunkBlock:
			p.toks, p.pos = c.toks, 0
			t := p.next()
			if t.kind != tokName {
				return nil, "", p.errorf("a block tag starts with %s", t)
			}
			switch t.s {
			case "endif", "elif", "else", "endfor", "endset", "endmacro", "endgeneration", "endfilter":
				return out, t.s, nil
			}
			n, err := p.statement(t.s)
			if err != nil {
				return nil, "", err
			}
			if n != nil {
				out = append(out, n)
			}
		}
	}
	return out, "", nil
}

func (p *parser) statement(name string) (node, error) {
	switch name {
	case "if":
		return p.ifStatement()
	case "for":
		return p.forStatement()
	case "set":
		return p.setStatement()
	case "macro":
		return p.macroStatement()
	case "break", "continue":
		if err := p.end(); err != nil {
			return nil, err
		}
		if name == "break" {
			return breakNode{}, nil
		}
		return continueNode{}, nil
	case "generation":
		// transformers' own tag, which marks what the assistant wrote for its
		// training masks. Rendering, it is its body and nothing else.
		if err := p.end(); err != nil {
			return nil, err
		}
		body, err := p.body("endgeneration")
		if err != nil {
			return nil, err
		}
		return ifNode{conds: []expr{constExpr{true}}, bodies: [][]node{body}}, nil
	}
	return nil, p.errorf("the %q tag is not implemented", name)
}

// body reads statements up to the one end tag that may close them.
func (p *parser) body(end string) ([]node, error) {
	line := p.line
	body, got, err := p.statements()
	if err != nil {
		return nil, err
	}
	if got != end {
		if got == "" {
			return nil, fmt.Errorf("line %d: the block is never closed with %s", line, end)
		}
		return nil, p.errorf("%s where %s was expected", got, end)
	}
	return body, p.end()
}

func (p *parser) ifStatement() (node, error) {
	n := ifNode{}
	for {
		cond, err := p.expression()
		if err != nil {
			return nil, err
		}
		if err := p.end(); err != nil {
			return nil, err
		}
		body, got, err := p.statements()
		if err != nil {
			return nil, err
		}
		n.conds = append(n.conds, cond)
		n.bodies = append(n.bodies, body)
		switch got {
		case "elif":
			continue
		case "else":
			if err := p.end(); err != nil {
				return nil, err
			}
			n.orElse, err = p.body("endif")
			return n, err
		case "endif":
			return n, p.end()
		case "":
			return nil, p.errorf("an if is never closed")
		}
		return nil, p.errorf("%s inside an if", got)
	}
}

func (p *parser) forStatement() (node, error) {
	n := forNode{}
	for {
		t := p.next()
		if t.kind != tokName {
			return nil, p.errorf("a loop variable cannot be %s", t)
		}
		n.targets = append(n.targets, t.s)
		if !p.skipOp(",") {
			break
		}
	}
	if !p.skipName("in") {
		return nil, p.errorf("a for without in")
	}
	var err error
	// The iterable stops short of a conditional expression, so that the `if`
	// that follows it filters the loop.
	if n.iter, err = p.or(); err != nil {
		return nil, err
	}
	if p.skipName("if") {
		if n.cond, err = p.expression(); err != nil {
			return nil, err
		}
	}
	if p.skipName("recursive") {
		return nil, p.errorf("recursive loops are not implemented")
	}
	if err := p.end(); err != nil {
		return nil, err
	}
	body, got, err := p.statements()
	if err != nil {
		return nil, err
	}
	n.body = body
	switch got {
	case "else":
		if err := p.end(); err != nil {
			return nil, err
		}
		n.orElse, err = p.body("endfor")
		return n, err
	case "endfor":
		return n, p.end()
	case "":
		return nil, p.errorf("a for is never closed")
	}
	return nil, p.errorf("%s inside a for", got)
}

func (p *parser) setStatement() (node, error) {
	var targets []string
	attr := ""
	for {
		t := p.next()
		if t.kind != tokName {
			return nil, p.errorf("cannot assign to %s", t)
		}
		targets = append(targets, t.s)
		if len(targets) == 1 && p.skipOp(".") {
			a := p.next()
			if a.kind != tokName {
				return nil, p.errorf("cannot assign to the attribute %s", a)
			}
			attr = a.s
			break
		}
		if !p.skipOp(",") {
			break
		}
	}
	if p.peek().kind == tokEOF {
		// {% set x %}...{% endset %}
		if len(targets) != 1 {
			return nil, p.errorf("a set block assigns one name")
		}
		p.next()
		body, err := p.body("endset")
		if err != nil {
			return nil, err
		}
		return setBlockNode{name: targets[0], attr: attr, body: body}, nil
	}
	if !p.skipOp("=") {
		return nil, p.errorf("a set without =")
	}
	e, err := p.tuple()
	if err != nil {
		return nil, err
	}
	return setNode{targets: targets, attr: attr, e: e}, p.end()
}

func (p *parser) macroStatement() (node, error) {
	t := p.next()
	if t.kind != tokName {
		return nil, p.errorf("a macro is named by %s", t)
	}
	n := macroNode{name: t.s}
	if !p.skipOp("(") {
		return nil, p.errorf("the macro %s has no parameter list", t.s)
	}
	for !p.skipOp(")") {
		if len(n.params) > 0 && !p.skipOp(",") {
			return nil, p.errorf("the parameters of %s are not separated by commas", t.s)
		}
		if p.skipOp(")") {
			break
		}
		a := p.next()
		if a.kind != tokName {
			return nil, p.errorf("a parameter cannot be %s", a)
		}
		var def expr
		if p.skipOp("=") {
			var err error
			if def, err = p.expression(); err != nil {
				return nil, err
			}
		}
		n.params = append(n.params, a.s)
		n.defaults = append(n.defaults, def)
	}
	if err := p.end(); err != nil {
		return nil, err
	}
	var err error
	n.body, err = p.body("endmacro")
	return n, err
}

// Tokens within one tag.

func (p *parser) peek() token { return p.toks[p.pos] }

func (p *parser) peekAt(n int) token {
	if p.pos+n < len(p.toks) {
		return p.toks[p.pos+n]
	}
	return p.toks[len(p.toks)-1]
}

func (p *parser) next() token {
	t := p.toks[p.pos]
	if t.kind != tokEOF {
		p.pos++
	}
	return t
}

func (p *parser) isOp(s string) bool {
	t := p.peek()
	return t.kind == tokOp && t.s == s
}

func (p *parser) isName(s string) bool {
	t := p.peek()
	return t.kind == tokName && t.s == s
}

func (p *parser) skipOp(s string) bool {
	if p.isOp(s) {
		p.pos++
		return true
	}
	return false
}

func (p *parser) skipName(s string) bool {
	if p.isName(s) {
		p.pos++
		return true
	}
	return false
}

func (p *parser) expect(s string) error {
	if !p.skipOp(s) {
		return p.errorf("expected %q, found %s", s, p.peek())
	}
	return nil
}

func (p *parser) end() error {
	if t := p.peek(); t.kind != tokEOF {
		return p.errorf("unexpected %s", t)
	}
	return nil
}

func (p *parser) errorf(format string, args ...any) error {
	line := p.line
	if p.pos < len(p.toks) {
		line = p.toks[p.pos].line
	}
	return fmt.Errorf("line %d: %s", line, fmt.Sprintf(format, args...))
}

// Expressions, loosest first.

// tuple reads `a, b` as a tuple, which only a set may assign.
func (p *parser) tuple() (expr, error) {
	e, err := p.expression()
	if err != nil || !p.isOp(",") {
		return e, err
	}
	items := []expr{e}
	for p.skipOp(",") {
		if p.peek().kind == tokEOF {
			break
		}
		e, err := p.expression()
		if err != nil {
			return nil, err
		}
		items = append(items, e)
	}
	return tupleExpr{items}, nil
}

func (p *parser) expression() (expr, error) {
	e, err := p.or()
	if err != nil {
		return nil, err
	}
	for p.skipName("if") {
		cond, err := p.or()
		if err != nil {
			return nil, err
		}
		var other expr
		if p.skipName("else") {
			if other, err = p.expression(); err != nil {
				return nil, err
			}
		}
		e = condExpr{cond: cond, a: e, b: other}
	}
	return e, nil
}

func (p *parser) or() (expr, error) {
	e, err := p.and()
	for err == nil && p.skipName("or") {
		var b expr
		b, err = p.and()
		e = orExpr{e, b}
	}
	return e, err
}

func (p *parser) and() (expr, error) {
	e, err := p.not()
	for err == nil && p.skipName("and") {
		var b expr
		b, err = p.not()
		e = andExpr{e, b}
	}
	return e, err
}

func (p *parser) not() (expr, error) {
	if p.skipName("not") {
		e, err := p.not()
		return notExpr{e}, err
	}
	return p.compare()
}

func (p *parser) compare() (expr, error) {
	line := p.peek().line
	first, err := p.math1()
	if err != nil {
		return nil, err
	}
	n := compareExpr{first: first, line: line}
	for {
		op := ""
		t := p.peek()
		switch {
		case t.kind == tokOp && (t.s == "==" || t.s == "!=" || t.s == "<" || t.s == ">" || t.s == "<=" || t.s == ">="):
			op = t.s
			p.next()
		case t.kind == tokName && t.s == "in":
			op = "in"
			p.next()
		case t.kind == tokName && t.s == "not" && p.peekAt(1).kind == tokName && p.peekAt(1).s == "in":
			op = "not in"
			p.next()
			p.next()
		}
		if op == "" {
			break
		}
		e, err := p.math1()
		if err != nil {
			return nil, err
		}
		n.ops = append(n.ops, op)
		n.rest = append(n.rest, e)
	}
	if len(n.ops) == 0 {
		return first, nil
	}
	return n, nil
}

func (p *parser) binary(ops []string, next func() (expr, error)) (expr, error) {
	e, err := next()
	for err == nil {
		t := p.peek()
		op := ""
		for _, o := range ops {
			if t.kind == tokOp && t.s == o {
				op = o
			}
		}
		if op == "" {
			break
		}
		p.next()
		var b expr
		b, err = next()
		e = binaryExpr{op: op, a: e, b: b, line: t.line}
	}
	return e, err
}

func (p *parser) math1() (expr, error)  { return p.binary([]string{"+", "-"}, p.concat) }
func (p *parser) concat() (expr, error) { return p.binary([]string{"~"}, p.math2) }
func (p *parser) math2() (expr, error) {
	return p.binary([]string{"*", "/", "//", "%"}, p.pow)
}
func (p *parser) pow() (expr, error) { return p.binary([]string{"**"}, p.unaryFiltered) }

func (p *parser) unaryFiltered() (expr, error) { return p.unary(true) }

func (p *parser) unary(filtered bool) (expr, error) {
	var e expr
	var err error
	switch {
	case p.isOp("-"), p.isOp("+"):
		op := p.next().s
		var inner expr
		if inner, err = p.unary(false); err != nil {
			return nil, err
		}
		e = unaryExpr{op: op, e: inner}
	default:
		if e, err = p.primary(); err != nil {
			return nil, err
		}
	}
	if e, err = p.postfix(e); err != nil {
		return nil, err
	}
	if filtered {
		return p.filters(e)
	}
	return e, nil
}

func (p *parser) primary() (expr, error) {
	t := p.next()
	switch t.kind {
	case tokName:
		switch t.s {
		case "true", "True":
			return constExpr{true}, nil
		case "false", "False":
			return constExpr{false}, nil
		case "none", "None":
			return constExpr{nil}, nil
		}
		return nameExpr{name: t.s, line: t.line}, nil
	case tokString:
		s := t.s
		for p.peek().kind == tokString {
			s += p.next().s
		}
		return constExpr{s}, nil
	case tokInt:
		return constExpr{t.i}, nil
	case tokFloat:
		return constExpr{t.f}, nil
	case tokOp:
		switch t.s {
		case "(":
			if p.skipOp(")") {
				return tupleExpr{}, nil
			}
			e, err := p.expression()
			if err != nil {
				return nil, err
			}
			if p.skipOp(")") {
				return e, nil
			}
			items := []expr{e}
			for p.skipOp(",") {
				if p.isOp(")") {
					break
				}
				e, err := p.expression()
				if err != nil {
					return nil, err
				}
				items = append(items, e)
			}
			return tupleExpr{items}, p.expect(")")
		case "[":
			var items []expr
			for !p.skipOp("]") {
				if len(items) > 0 {
					if err := p.expect(","); err != nil {
						return nil, err
					}
					if p.skipOp("]") {
						break
					}
				}
				e, err := p.expression()
				if err != nil {
					return nil, err
				}
				items = append(items, e)
			}
			return listExpr{items}, nil
		case "{":
			var n dictExpr
			for !p.skipOp("}") {
				if len(n.keys) > 0 {
					if err := p.expect(","); err != nil {
						return nil, err
					}
					if p.skipOp("}") {
						break
					}
				}
				k, err := p.expression()
				if err != nil {
					return nil, err
				}
				if err := p.expect(":"); err != nil {
					return nil, err
				}
				v, err := p.expression()
				if err != nil {
					return nil, err
				}
				n.keys = append(n.keys, k)
				n.vals = append(n.vals, v)
			}
			return n, nil
		}
	}
	return nil, p.errorf("unexpected %s", t)
}

// postfix reads attributes, subscripts and calls.
func (p *parser) postfix(e expr) (expr, error) {
	for {
		switch {
		case p.skipOp("."):
			t := p.next()
			switch t.kind {
			case tokName:
				e = attrExpr{obj: e, name: t.s}
			case tokInt:
				e = itemExpr{obj: e, key: constExpr{t.i}}
			default:
				return nil, p.errorf("an attribute cannot be %s", t)
			}
		case p.skipOp("["):
			var err error
			if e, err = p.subscript(e); err != nil {
				return nil, err
			}
		case p.isOp("("):
			args, kwargs, err := p.arguments()
			if err != nil {
				return nil, err
			}
			e = callExpr{fn: e, args: args, kwargs: kwargs}
		default:
			return e, nil
		}
	}
}

func (p *parser) subscript(obj expr) (expr, error) {
	var parts [3]expr
	slice := false
	for i := 0; i < 3; i++ {
		if !p.isOp(":") && !p.isOp("]") {
			e, err := p.expression()
			if err != nil {
				return nil, err
			}
			parts[i] = e
		}
		if i < 2 && p.skipOp(":") {
			slice = true
			continue
		}
		break
	}
	if err := p.expect("]"); err != nil {
		return nil, err
	}
	if !slice {
		return itemExpr{obj: obj, key: parts[0]}, nil
	}
	return sliceExpr{obj: obj, lo: parts[0], hi: parts[1], step: parts[2]}, nil
}

// arguments reads a parenthesised argument list, the opening parenthesis
// included.
func (p *parser) arguments() ([]expr, []kwarg, error) {
	if err := p.expect("("); err != nil {
		return nil, nil, err
	}
	var args []expr
	var kwargs []kwarg
	for !p.skipOp(")") {
		if len(args)+len(kwargs) > 0 {
			if err := p.expect(","); err != nil {
				return nil, nil, err
			}
			if p.skipOp(")") {
				break
			}
		}
		if t := p.peek(); t.kind == tokName && p.peekAt(1).kind == tokOp && p.peekAt(1).s == "=" {
			p.next()
			p.next()
			e, err := p.expression()
			if err != nil {
				return nil, nil, err
			}
			kwargs = append(kwargs, kwarg{name: t.s, e: e})
			continue
		}
		if len(kwargs) > 0 {
			return nil, nil, p.errorf("a positional argument after a keyword one")
		}
		e, err := p.expression()
		if err != nil {
			return nil, nil, err
		}
		args = append(args, e)
	}
	return args, kwargs, nil
}

// filters reads `| name(args)` and `is [not] name args`, which may follow one
// another in any order.
func (p *parser) filters(e expr) (expr, error) {
	for {
		switch {
		case p.skipOp("|"):
			t := p.next()
			if t.kind != tokName {
				return nil, p.errorf("a filter cannot be %s", t)
			}
			name := t.s
			for p.isOp(".") && p.peekAt(1).kind == tokName {
				p.next()
				name += "." + p.next().s
			}
			f := filterExpr{e: e, name: name, line: t.line}
			if p.isOp("(") {
				var err error
				if f.args, f.kwargs, err = p.arguments(); err != nil {
					return nil, err
				}
			}
			var err error
			if e, err = p.postfix(f); err != nil {
				return nil, err
			}
		case p.skipName("is"):
			negate := p.skipName("not")
			t := p.next()
			if t.kind != tokName {
				return nil, p.errorf("a test cannot be %s", t)
			}
			test := testExpr{e: e, name: t.s, negate: negate, line: t.line}
			if p.isOp("(") {
				args, kwargs, err := p.arguments()
				if err != nil {
					return nil, err
				}
				if len(kwargs) > 0 {
					return nil, p.errorf("the test %s takes no keyword arguments", t.s)
				}
				test.args = args
			} else if p.startsArgument() {
				// `is divisibleby 3`: one argument, no parentheses.
				arg, err := p.unary(false)
				if err != nil {
					return nil, err
				}
				test.args = []expr{arg}
			}
			e = test
		default:
			return e, nil
		}
	}
}

// startsArgument says whether what follows a test's name is its argument
// rather than the rest of the expression.
func (p *parser) startsArgument() bool {
	t := p.peek()
	switch t.kind {
	case tokString, tokInt, tokFloat:
		return true
	case tokName:
		switch t.s {
		case "else", "or", "and", "not", "in", "is", "if":
			return false
		}
		return true
	case tokOp:
		return t.s == "[" || t.s == "{"
	}
	return false
}
