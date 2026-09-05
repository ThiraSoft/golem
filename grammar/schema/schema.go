// Package schema turns a JSON Schema into a GBNF grammar.
//
// It is a port of llama.cpp's common/json-schema-to-grammar.cpp, down to the
// names it gives the rules it generates and the order it prints them in: the
// two are compared string for string in the tests, because a grammar that
// differs by a rule name is a grammar nobody can check against the reference.
//
// What it refuses rather than approximates: a pattern, which is a regular
// expression and a second port; a numeric bound, which llama.cpp compiles into
// a digit-by-digit grammar; and a $ref that points somewhere else on the
// network. A schema that asks for one of those is refused by name, because a
// constraint silently dropped is worse than a request refused.
package schema

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// spaceRule is what llama.cpp allows between two structural elements: nothing,
// one space, or a newline and a little indentation.
//
// It matters more than it looks. A BPE vocabulary holds tokens like `": "` and
// `",\n    "`; a grammar that demanded a single space would refuse all of them
// and force the model onto one-character tokens, which costs about two thirds
// of the generation rate.
const spaceRule = `| " " | "\n"{1,2} [ \t]{0,20}`

type builtin struct {
	content string
	deps    []string
}

var primitives = map[string]builtin{
	"boolean":       {`("true" | "false") space`, nil},
	"decimal-part":  {`[0-9]{1,16}`, nil},
	"integral-part": {`[0] | [1-9] [0-9]{0,15}`, nil},
	"number":        {`("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space`, []string{"integral-part", "decimal-part"}},
	"integer":       {`("-"? integral-part) space`, []string{"integral-part"}},
	"value":         {`object | array | string | number | boolean | null`, []string{"object", "array", "string", "number", "boolean", "null"}},
	"object":        {`"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space`, []string{"string", "value"}},
	"array":         {`"[" space ( value ("," space value)* )? "]" space`, []string{"value"}},
	"uuid":          {`"\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"" space`, nil},
	"char":          {`[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})`, nil},
	"string":        {`"\"" char* "\"" space`, []string{"char"}},
	"null":          {`"null" space`, nil},
}

var stringFormats = map[string]builtin{
	"date":             {`[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )`, nil},
	"time":             {`([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )`, nil},
	"date-time":        {`date "T" time`, []string{"date", "time"}},
	"date-string":      {`"\"" date "\"" space`, []string{"date"}},
	"time-string":      {`"\"" time "\"" space`, []string{"time"}},
	"date-time-string": {`"\"" date-time "\"" space`, []string{"date-time"}},
}

func reserved(name string) bool {
	if name == "root" {
		return true
	}
	_, ok := primitives[name]
	if ok {
		return true
	}
	_, ok = stringFormats[name]
	return ok
}

var (
	invalidRuleChars = regexp.MustCompile(`[^a-zA-Z0-9-]+`)
	uuidFormat       = regexp.MustCompile(`^uuid[1-5]?$`)
)

// ToGBNF compiles a JSON Schema into a grammar. The rule it starts at is root,
// as everywhere else here.
func ToGBNF(raw []byte) (string, error) {
	doc, err := parse(raw)
	if err != nil {
		return "", fmt.Errorf("schema: this is not JSON: %w", err)
	}
	c := &converter{rules: map[string]string{"space": spaceRule}, refs: map[string]*value{}}
	if err := c.resolveRefs(doc, doc); err != nil {
		return "", err
	}
	c.visit(doc, "")
	if len(c.errs) > 0 {
		return "", fmt.Errorf("schema: %s", strings.Join(c.errs, "; "))
	}
	return c.format(), nil
}

// JSON is the grammar of any JSON value at all, which is what a request asking
// for an object and naming no schema gets.
func JSON() string {
	c := &converter{rules: map[string]string{"space": spaceRule}, refs: map[string]*value{}}
	c.addRule("root", c.addPrimitive("object", primitives["object"]))
	return c.format()
}

type converter struct {
	rules     map[string]string
	refs      map[string]*value
	resolving map[string]bool
	errs      []string
}

func (c *converter) fail(format string, args ...any) string {
	c.errs = append(c.errs, fmt.Sprintf(format, args...))
	return ""
}

func (c *converter) format() string {
	names := make([]string, 0, len(c.rules))
	for name := range c.rules {
		names = append(names, name)
	}
	sort.Strings(names)
	var b strings.Builder
	for _, name := range names {
		fmt.Fprintf(&b, "%s ::= %s\n", name, c.rules[name])
	}
	return b.String()
}

// addRule names a rule, and gives it a number if the name is taken by a
// different one.
func (c *converter) addRule(name, rule string) string {
	esc := invalidRuleChars.ReplaceAllString(name, "-")
	if have, ok := c.rules[esc]; !ok || have == rule {
		c.rules[esc] = rule
		return esc
	}
	for i := 0; ; i++ {
		key := esc + strconv.Itoa(i)
		if have, ok := c.rules[key]; !ok || have == rule {
			c.rules[key] = rule
			return key
		}
	}
}

func (c *converter) addPrimitive(name string, rule builtin) string {
	n := c.addRule(name, rule.content)
	for _, dep := range rule.deps {
		b, ok := primitives[dep]
		if !ok {
			if b, ok = stringFormats[dep]; !ok {
				return c.fail("the rule %q is not one this converter knows", dep)
			}
		}
		if _, have := c.rules[dep]; !have {
			c.addPrimitive(dep, b)
		}
	}
	return n
}

// resolveRefs walks the schema and records what every local $ref points at. A
// reference to another document would mean fetching it, and a converter that
// fetches what a request names is a converter that can be aimed.
func (c *converter) resolveRefs(node, root *value) error {
	if node == nil {
		return nil
	}
	switch node.kind {
	case kindArray:
		for _, item := range node.array {
			if err := c.resolveRefs(item, root); err != nil {
				return err
			}
		}
	case kindObject:
		if ref := node.get("$ref"); ref.isString() {
			if _, have := c.refs[ref.text]; have {
				return nil
			}
			if !strings.HasPrefix(ref.text, "#/") {
				return fmt.Errorf("schema: $ref %q points outside this document, which this server does not follow", ref.text)
			}
			target := root
			for _, step := range strings.Split(strings.TrimPrefix(ref.text, "#/"), "/") {
				step = strings.ReplaceAll(strings.ReplaceAll(step, "~1", "/"), "~0", "~")
				switch {
				case target.isObject() && target.has(step):
					target = target.get(step)
				case target.isArray():
					i, err := strconv.Atoi(step)
					if err != nil || i < 0 || i >= len(target.array) {
						return fmt.Errorf("schema: $ref %q names %q, which is not there", ref.text, step)
					}
					target = target.array[i]
				default:
					return fmt.Errorf("schema: $ref %q names %q, which is not there", ref.text, step)
				}
			}
			c.refs[ref.text] = target
			return c.resolveRefs(target, root)
		}
		for _, m := range node.members {
			if err := c.resolveRefs(m.value, root); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c *converter) resolveRef(ref string) string {
	fragment := ref
	if at := strings.Index(ref, "#"); at >= 0 {
		fragment = ref[at+1:]
	}
	name := "ref" + invalidRuleChars.ReplaceAllString(fragment, "-")
	if c.resolving == nil {
		c.resolving = map[string]bool{}
	}
	if _, have := c.rules[name]; !have && !c.resolving[ref] {
		c.resolving[ref] = true
		name = c.visit(c.refs[ref], name)
		delete(c.resolving, ref)
	}
	return name
}

// visit is the whole conversion: one schema in, the name of the rule that
// matches it out.
func (c *converter) visit(s *value, name string) string {
	schemaType := s.get("type")
	format := ""
	if f := s.get("format"); f.isString() {
		format = f.text
	}
	ruleName := name
	switch {
	case reserved(name):
		ruleName = name + "-"
	case name == "":
		ruleName = "root"
	}

	if ref := s.get("$ref"); ref.isString() {
		return c.addRule(ruleName, c.resolveRef(ref.text))
	}
	if alts := s.get("oneOf"); alts.isArray() {
		return c.addRule(ruleName, c.union(name, alts.array))
	}
	if alts := s.get("anyOf"); alts.isArray() {
		return c.addRule(ruleName, c.union(name, alts.array))
	}
	if schemaType.isArray() {
		// A type given as a list is a union of the same schema under each.
		var each []*value
		for _, t := range schemaType.array {
			copy := &value{kind: kindObject}
			for _, m := range s.members {
				if m.name == "type" {
					copy.members = append(copy.members, memberOf("type", t))
					continue
				}
				copy.members = append(copy.members, m)
			}
			each = append(each, copy)
		}
		return c.addRule(ruleName, c.union(name, each))
	}
	if k := s.get("const"); k != nil {
		return c.addRule(ruleName, literal(k.dump())+" space")
	}
	if e := s.get("enum"); e.isArray() {
		alts := make([]string, 0, len(e.array))
		for _, v := range e.array {
			alts = append(alts, literal(v.dump()))
		}
		return c.addRule(ruleName, "("+strings.Join(alts, " | ")+") space")
	}

	isType := func(want string) bool {
		return schemaType == nil || (schemaType.isString() && schemaType.text == want)
	}

	if isType("object") && (s.has("properties") ||
		(s.has("additionalProperties") && !s.get("additionalProperties").isTrue())) {
		required := map[string]bool{}
		if r := s.get("required"); r.isArray() {
			for _, item := range r.array {
				if item.isString() {
					required[item.text] = true
				}
			}
		}
		var props []member
		if p := s.get("properties"); p.isObject() {
			props = p.members
		}
		return c.addRule(ruleName, c.object(props, required, name, s.get("additionalProperties")))
	}
	if (isType("object") || isType("string")) && s.get("allOf").isArray() {
		return c.addRule(ruleName, c.allOf(s.get("allOf"), name))
	}
	if isType("array") && (s.has("items") || s.has("prefixItems")) {
		items := s.get("items")
		if items == nil {
			items = s.get("prefixItems")
		}
		if items.isArray() {
			// A tuple: each position has a schema of its own.
			parts := make([]string, 0, len(items.array))
			for i, item := range items.array {
				parts = append(parts, c.visit(item, join(name, "tuple-"+strconv.Itoa(i))))
			}
			return c.addRule(ruleName, `"[" space `+strings.Join(parts, ` "," space `)+` "]" space`)
		}
		item := c.visit(items, join(name, "item"))
		min := s.get("minItems").integer(0)
		max := -1
		if m := s.get("maxItems"); m != nil {
			max = m.integer(-1)
		}
		return c.addRule(ruleName, `"[" space `+repetition(item, min, max, `"," space`)+` "]" space`)
	}
	if isType("string") && s.has("pattern") {
		return c.fail("this server does not compile a schema's pattern into a grammar")
	}
	if isType("string") && uuidFormat.MatchString(format) {
		if ruleName == "root" {
			return c.addPrimitive("root", primitives["uuid"])
		}
		return c.addPrimitive(format, primitives["uuid"])
	}
	if isType("string") {
		if b, ok := stringFormats[format+"-string"]; ok {
			return c.addRule(ruleName, c.addPrimitive(format+"-string", b))
		}
	}
	if schemaType.isString() && schemaType.text == "string" && (s.has("minLength") || s.has("maxLength")) {
		char := c.addPrimitive("char", primitives["char"])
		min := s.get("minLength").integer(0)
		max := -1
		if m := s.get("maxLength"); m != nil {
			max = m.integer(-1)
		}
		return c.addRule(ruleName, `"\"" `+repetition(char, min, max, "")+` "\"" space`)
	}
	for _, bound := range []string{"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"} {
		if s.has(bound) {
			return c.fail("this server does not compile a schema's %s into a grammar", bound)
		}
	}
	if s.isObject() && len(s.members) == 0 {
		return c.addRule(ruleName, c.addPrimitive("object", primitives["object"]))
	}
	if schemaType.isString() && schemaType.text == "object" {
		return c.addRule(ruleName, c.addPrimitive("object", primitives["object"]))
	}
	if schemaType == nil && s.isObject() {
		// No type and no keyword that says anything about the shape — a
		// description, say. That is every value.
		return c.addRule(ruleName, c.addPrimitive("value", primitives["value"]))
	}
	if !schemaType.isString() {
		return c.fail("a schema this converter cannot read: %s", s.dump())
	}
	b, ok := primitives[schemaType.text]
	if !ok {
		return c.fail("a schema of type %q, which is not a JSON type", schemaType.text)
	}
	if ruleName == "root" {
		return c.addPrimitive("root", b)
	}
	return c.addPrimitive(schemaType.text, b)
}

func (c *converter) union(name string, alts []*value) string {
	parts := make([]string, 0, len(alts))
	for i, alt := range alts {
		sub := name + "-" + strconv.Itoa(i)
		if name == "" {
			sub = "alternative-" + strconv.Itoa(i)
		}
		parts = append(parts, c.visit(alt, sub))
	}
	return strings.Join(parts, " | ")
}

// allOf folds the components into one object, which is what llama.cpp does: the
// properties of every component, required unless the component arrived through
// an anyOf, and the intersection of the enumerations if they all name one.
func (c *converter) allOf(all *value, name string) string {
	required := map[string]bool{}
	var props []member
	enums := map[string]int{}

	var add func(comp *value, isRequired bool)
	add = func(comp *value, isRequired bool) {
		if ref := comp.get("$ref"); ref.isString() {
			add(c.refs[ref.text], isRequired)
			return
		}
		if p := comp.get("properties"); p.isObject() {
			for _, m := range p.members {
				props = append(props, m)
				if isRequired {
					required[m.name] = true
				}
			}
			return
		}
		if e := comp.get("enum"); e.isArray() {
			for _, v := range e.array {
				enums[literal(v.dump())]++
			}
		}
	}
	for _, comp := range all.array {
		if any := comp.get("anyOf"); any.isArray() {
			for _, alt := range any.array {
				add(alt, false)
			}
			continue
		}
		add(comp, true)
	}
	if len(enums) > 0 {
		var shared []string
		for rule, n := range enums {
			if n == len(all.array) {
				shared = append(shared, rule)
			}
		}
		if len(shared) > 0 {
			sort.Strings(shared)
			return "(" + strings.Join(shared, " | ") + ") space"
		}
	}
	return c.object(props, required, name, nil)
}

// object builds the rule of an object: its required properties in order, then
// whatever optional ones may follow, each tail a rule of its own so that a
// property may be left out without the alternatives multiplying.
func (c *converter) object(props []member, required map[string]bool, name string, additional *value) string {
	var requiredProps, optionalProps, names []string
	kv := map[string]string{}

	for _, p := range props {
		rule := c.visit(p.value, join(name, p.name))
		kv[p.name] = c.addRule(join(name, p.name)+"-kv",
			literal(quote(p.name))+` space ":" space `+rule)
		if required[p.name] {
			requiredProps = append(requiredProps, p.name)
		} else {
			optionalProps = append(optionalProps, p.name)
		}
		names = append(names, p.name)
	}

	if additional.isTrue() || additional.isObject() {
		sub := join(name, "additional")
		valueRule := ""
		if additional.isObject() {
			valueRule = c.visit(additional, sub+"-value")
		} else {
			valueRule = c.addPrimitive("value", primitives["value"])
		}
		keyRule := ""
		if len(names) > 0 {
			keyRule = c.addRule(sub+"-k", c.notStrings(names))
		} else {
			keyRule = c.addPrimitive("string", primitives["string"])
		}
		kv["*"] = c.addRule(sub+"-kv", keyRule+` ":" space `+valueRule)
		optionalProps = append(optionalProps, "*")
	}

	rule := `"{" space `
	for i, p := range requiredProps {
		if i > 0 {
			rule += ` "," space `
		}
		rule += kv[p]
	}

	if len(optionalProps) > 0 {
		rule += " ("
		if len(requiredProps) > 0 {
			rule += ` "," space ( `
		}
		var tail func(ks []string, firstOptional bool) string
		tail = func(ks []string, firstOptional bool) string {
			if len(ks) == 0 {
				return ""
			}
			k := ks[0]
			comma := `( "," space ` + kv[k] + " )"
			var res string
			if firstOptional {
				if k == "*" {
					res = comma + "*"
				} else {
					res = comma + "?"
				}
			} else if k == "*" {
				res = kv[k] + " " + comma + "*"
			} else {
				res = kv[k]
			}
			if len(ks) > 1 {
				res += " " + c.addRule(join(name, k)+"-rest", tail(ks[1:], true))
			}
			return res
		}
		for i := range optionalProps {
			if i > 0 {
				rule += " | "
			}
			rule += tail(optionalProps[i:], false)
		}
		if len(requiredProps) > 0 {
			rule += " )"
		}
		rule += " )?"
	}
	return rule + ` "}" space`
}

// notStrings is a string that is none of the named ones, which is what the key
// of an extra property has to be when the named ones have rules of their own.
// The strings go into a trie, and the trie is read out as alternatives.
func (c *converter) notStrings(strings_ []string) string {
	type node struct {
		children map[byte]*node
		keys     []byte
		end      bool
	}
	root := &node{children: map[byte]*node{}}
	for _, s := range strings_ {
		at := root
		for i := 0; i < len(s); i++ {
			next, ok := at.children[s[i]]
			if !ok {
				next = &node{children: map[byte]*node{}}
				at.children[s[i]] = next
				at.keys = append(at.keys, s[i])
			}
			at = next
		}
		at.end = true
	}

	char := c.addPrimitive("char", primitives["char"])
	var b strings.Builder
	b.WriteString(`["] ( `)
	var visit func(n *node)
	visit = func(n *node) {
		keys := append([]byte(nil), n.keys...)
		sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
		var rejects strings.Builder
		first := true
		for _, k := range keys {
			rejects.WriteByte(k)
			if first {
				first = false
			} else {
				b.WriteString(" | ")
			}
			fmt.Fprintf(&b, "[%c]", k)
			child := n.children[k]
			if len(child.children) > 0 {
				b.WriteString(" (")
				visit(child)
				b.WriteString(")")
			} else if child.end {
				b.WriteString(" " + char + "+")
			}
		}
		if len(n.children) > 0 {
			if !first {
				b.WriteString(" | ")
			}
			fmt.Fprintf(&b, `[^"%s] %s*`, rejects.String(), char)
		}
	}
	visit(root)
	b.WriteString(" )")
	if !root.end {
		b.WriteString("?")
	}
	b.WriteString(` ["] space`)
	return b.String()
}

// repetition writes `item` repeated between min and max times, with max below
// zero meaning no bound, and a separator between two of them where there is one.
func repetition(item string, min, max int, separator string) string {
	if max == 0 {
		return ""
	}
	if min == 0 && max == 1 {
		return item + "?"
	}
	if separator == "" {
		switch {
		case min == 1 && max < 0:
			return item + "+"
		case min == 0 && max < 0:
			return item + "*"
		}
		bound := ""
		if max >= 0 {
			bound = strconv.Itoa(max)
		}
		return item + "{" + strconv.Itoa(min) + "," + bound + "}"
	}
	inner := min - 1
	if min == 0 {
		inner = 0
	}
	outer := max
	if max >= 0 {
		outer = max - 1
	}
	res := item + " " + repetition("("+separator+" "+item+")", inner, outer, "")
	if min == 0 {
		res = "(" + res + ")?"
	}
	return res
}

// literal writes a piece of text as a GBNF string.
func literal(text string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range text {
		switch r {
		case '\r':
			b.WriteString(`\r`)
		case '\n':
			b.WriteString(`\n`)
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// quote is a name as the JSON text of a string, which is what the literal of a
// property's key has to match — the quotes included.
func quote(name string) string {
	var b strings.Builder
	writeJSONString(&b, name)
	return b.String()
}

func join(name, part string) string {
	if name == "" {
		return part
	}
	return name + "-" + part
}
