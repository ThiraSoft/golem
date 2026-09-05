package schema

import (
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/grammar"
)

// The cases and the grammars expected of them are llama.cpp's own
// (tests/test-json-schema-to-grammar.cpp), compared line for line. A rule named
// differently is a different grammar to anyone checking one against the other.

func check(t *testing.T, name, in, want string) {
	t.Helper()
	got, err := ToGBNF([]byte(in))
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if trim(got) != trim(want) {
		t.Fatalf("%s:\n--- produced ---\n%s\n--- expected ---\n%s", name, trim(got), trim(want))
	}
	// Whatever comes out has to be a grammar the engine can read.
	if _, err := grammar.Parse(got); err != nil {
		t.Fatalf("%s: the grammar produced does not parse: %v\n%s", name, err, got)
	}
}

func trim(s string) string {
	var out []string
	for _, line := range strings.Split(s, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			out = append(out, line)
		}
	}
	return strings.Join(out, "\n")
}

func TestPrimitives(t *testing.T) {
	check(t, "boolean", `{"type": "boolean"}`, `
		root ::= ("true" | "false") space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)

	check(t, "integer", `{"type": "integer"}`, `
		integral-part ::= [0] | [1-9] [0-9]{0,15}
		root ::= ("-"? integral-part) space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)

	check(t, "string", `{"type": "string"}`, `
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		root ::= "\"" char* "\"" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)
}

func TestConstAndEnum(t *testing.T) {
	check(t, "string const", `{"const": "foo"}`, `
		root ::= "\"foo\"" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)

	check(t, "non-string const", `{"const": 123}`, `
		root ::= "123" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)

	check(t, "non-string enum", `{"enum": ["red", "amber", "green", null, 42, ["foo"]]}`, `
		root ::= ("\"red\"" | "\"amber\"" | "\"green\"" | "null" | "42" | "[\"foo\"]") space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)
}

func TestStringLengths(t *testing.T) {
	check(t, "min length 1", `{"type": "string", "minLength": 1}`, `
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		root ::= "\"" char+ "\"" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)

	check(t, "min length 3", `{"type": "string", "minLength": 3}`, `
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		root ::= "\"" char{3,} "\"" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)

	check(t, "min and max length", `{"type": "string", "minLength": 1, "maxLength": 4}`, `
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		root ::= "\"" char{1,4} "\"" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)
}

func TestArrays(t *testing.T) {
	check(t, "string array", `{"type": "array", "prefixItems": {"type": "string"}}`, `
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		root ::= "[" space (string ("," space string)*)? "]" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}
		string ::= "\"" char* "\"" space`)

	check(t, "tuple", `{"prefixItems": [{"type": "string"}, {"type": "number"}]}`, `
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		decimal-part ::= [0-9]{1,16}
		integral-part ::= [0] | [1-9] [0-9]{0,15}
		number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
		root ::= "[" space string "," space number "]" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}
		string ::= "\"" char* "\"" space`)

	check(t, "nullable string array", `{"type": ["array", "null"], "prefixItems": {"type": "string"}}`, `
		alternative-0 ::= "[" space (string ("," space string)*)? "]" space
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		null ::= "null" space
		root ::= alternative-0 | null
		space ::= | " " | "\n"{1,2} [ \t]{0,20}
		string ::= "\"" char* "\"" space`)
}

func TestObjects(t *testing.T) {
	check(t, "required and additional", `{
		"type": "object",
		"properties": {"a": {"type": "number"}},
		"required": ["a"],
		"additionalProperties": {"type": "string"}
	}`, `
		a-kv ::= "\"a\"" space ":" space number
		additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["] space
		additional-kv ::= additional-k ":" space string
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		decimal-part ::= [0-9]{1,16}
		integral-part ::= [0] | [1-9] [0-9]{0,15}
		number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
		root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? "}" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}
		string ::= "\"" char* "\"" space`)

	check(t, "optional and additional", `{
		"type": "object",
		"properties": {"a": {"type": "number"}},
		"additionalProperties": {"type": "number"}
	}`, `
		a-kv ::= "\"a\"" space ":" space number
		a-rest ::= ( "," space additional-kv )*
		additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["] space
		additional-kv ::= additional-k ":" space number
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		decimal-part ::= [0-9]{1,16}
		integral-part ::= [0] | [1-9] [0-9]{0,15}
		number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
		root ::= "{" space  (a-kv a-rest | additional-kv ( "," space additional-kv )* )? "}" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)
}

func TestReferences(t *testing.T) {
	check(t, "top-level ref", `{
		"$ref": "#/definitions/foo",
		"definitions": {
			"foo": {
				"type": "object",
				"properties": {"a": {"type": "string"}},
				"required": ["a"],
				"additionalProperties": false
			}
		}
	}`, `
		char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
		ref-definitions-foo ::= "{" space ref-definitions-foo-a-kv "}" space
		ref-definitions-foo-a-kv ::= "\"a\"" space ":" space string
		root ::= ref-definitions-foo
		space ::= | " " | "\n"{1,2} [ \t]{0,20}
		string ::= "\"" char* "\"" space`)
}

func TestUnions(t *testing.T) {
	check(t, "anyOf of refs", `{
		"anyOf": [{"$ref": "#/definitions/foo"}, {"$ref": "#/definitions/bar"}],
		"definitions": {
			"foo": {"properties": {"a": {"type": "number"}}},
			"bar": {"properties": {"b": {"type": "number"}}}
		},
		"type": "object"
	}`, `
		alternative-0 ::= ref-definitions-foo
		alternative-1 ::= ref-definitions-bar
		decimal-part ::= [0-9]{1,16}
		integral-part ::= [0] | [1-9] [0-9]{0,15}
		number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
		ref-definitions-bar ::= "{" space  (ref-definitions-bar-b-kv )? "}" space
		ref-definitions-bar-b-kv ::= "\"b\"" space ":" space number
		ref-definitions-foo ::= "{" space  (ref-definitions-foo-a-kv )? "}" space
		ref-definitions-foo-a-kv ::= "\"a\"" space ":" space number
		root ::= alternative-0 | alternative-1
		space ::= | " " | "\n"{1,2} [ \t]{0,20}`)
}

func TestStringFormats(t *testing.T) {
	check(t, "exotic formats", `{
		"items": [
			{"format": "date"},
			{"format": "uuid"},
			{"format": "time"},
			{"format": "date-time"}
		]
	}`, `
		date ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )
		date-string ::= "\"" date "\"" space
		date-time ::= date "T" time
		date-time-string ::= "\"" date-time "\"" space
		root ::= "[" space tuple-0 "," space uuid "," space tuple-2 "," space tuple-3 "]" space
		space ::= | " " | "\n"{1,2} [ \t]{0,20}
		time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
		time-string ::= "\"" time "\"" space
		tuple-0 ::= date-string
		tuple-2 ::= time-string
		tuple-3 ::= date-time-string
		uuid ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"" space`)
}

func TestTheGenericObjectGrammar(t *testing.T) {
	if _, err := grammar.Parse(JSON()); err != nil {
		t.Fatalf("the grammar of any object does not parse: %v", err)
	}
	if !strings.Contains(JSON(), "root ::= object") {
		t.Fatalf("the grammar of any object does not start at one:\n%s", JSON())
	}
}

// What is refused by name rather than approximated. A constraint silently
// dropped is worse than a request refused: the answer comes back looking right
// and is not.
func TestWhatIsRefused(t *testing.T) {
	for _, c := range []struct{ in, says string }{
		{`{"type": "string", "pattern": "^a+$"}`, "pattern"},
		{`{"type": "integer", "minimum": 3}`, "minimum"},
		{`{"type": "integer", "exclusiveMaximum": 3}`, "exclusiveMaximum"},
		{`{"$ref": "https://example.com/schema.json"}`, "outside this document"},
		{`{"$ref": "#/nowhere"}`, "not there"},
		{`{"type": "colour"}`, "not a JSON type"},
		{`{"type":`, "not JSON"},
	} {
		_, err := ToGBNF([]byte(c.in))
		if err == nil {
			t.Fatalf("%s was accepted, and should not have been", c.in)
		}
		if !strings.Contains(err.Error(), c.says) {
			t.Fatalf("%s was refused with %q, which does not say %q", c.in, err, c.says)
		}
	}
}

// The two halves together: a schema compiled to a grammar, and a document
// walked through it byte by byte. The tokenizer here is the cruellest one — a
// token per byte — because that is what makes the automaton do the most work.
func TestADocumentWalksThroughACompiledSchema(t *testing.T) {
	src, err := ToGBNF([]byte(`{
		"type": "object",
		"properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
		"required": ["city", "population"],
		"additionalProperties": false
	}`))
	if err != nil {
		t.Fatal(err)
	}
	rules, err := grammar.Parse(src)
	if err != nil {
		t.Fatal(err)
	}

	// One token per byte, plus one that ends the turn.
	const eog = 256
	toks := grammar.NewTokens(eog+1,
		func(id int32) string {
			if int(id) == eog {
				return "<eos>"
			}
			return string([]byte{byte(id)})
		},
		func(id int32) bool { return int(id) == eog })

	walk := func(doc string) (int, bool) {
		g := grammar.New(rules, toks)
		for i := 0; i < len(doc); i++ {
			id := int32(doc[i])
			if !g.Allows(id) {
				return i, false
			}
			g.Accept(id)
		}
		return len(doc), g.Allows(eog)
	}

	good := `{"city": "Lyon", "population": 520000}`
	if at, done := walk(good); !done {
		t.Fatalf("a document the schema describes stopped at byte %d of %q", at, good)
	}

	for _, bad := range []string{
		`{"city": "Lyon"}`,                             // a required property missing
		`{"city": "Lyon", "population": "520000"}`,     // an integer written as a string
		`{"city": "Lyon", "population": 5, "x": true}`, // a property the schema forbids
		`{"population": 5, "city": "Lyon"}`,            // the properties out of order
	} {
		if _, done := walk(bad); done {
			t.Fatalf("the grammar accepted %q and the schema does not describe it", bad)
		}
	}
}
