# grammar/schema

A JSON Schema in, a GBNF grammar out.

```go
src, err := schema.ToGBNF([]byte(`{"type":"object","properties":{"n":{"type":"integer"}}}`))
rules, err := grammar.Parse(src)
```

`JSON()` is the grammar of any JSON value at all, which is what a request asking
for an object and naming no schema gets.

A port of `common/json-schema-to-grammar.cpp`, down to the names it gives the
rules it generates and the order it prints them in. The tests compare what golem
produces against llama.cpp's own expected grammars, line for line — which only
works if the fidelity goes that far.

Two things could not be copied. JSON arrives through `encoding/json`, which
decodes an object into a map and loses the order of its members, and the order of
an object's properties decides the order of the rules; so the schema is read from
the token stream and keeps what it saw. And the literals go through an escaper of
our own, because `encoding/json` also escapes `<`, `>` and `&`, which would put a
different literal in the grammar.

Supported: `type` in its six forms and as a list, `properties`, `required`,
`additionalProperties`, `items`, `prefixItems`, `minItems`, `maxItems`,
`minLength`, `maxLength`, `enum`, `const`, `anyOf`, `oneOf`, `allOf`, `$ref`
into the same document, and the string formats `date`, `time`, `date-time` and
`uuid`.

Refused by name: `pattern`, which is a regular expression and a port of its own;
`minimum` and its three relatives, which llama.cpp compiles into a
digit-by-digit grammar; and a `$ref` that points at another document, which
would make a converter fetch what a request names. A constraint silently dropped
is worse than a request refused — the answer comes back looking right and is
not.

The whitespace rule is llama.cpp's, and it is not cosmetic. A BPE vocabulary
holds tokens like `": "` and `",\n    "`; a grammar that demanded a single space
would refuse all of them and force the model onto one-character tokens.
