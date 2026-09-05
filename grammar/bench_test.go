package grammar

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// A vocabulary the size of a real one, of pieces that look like a tokenizer's:
// mostly words, some punctuation, some fragments.
func bigVocabulary(n int) *Tokens {
	rng := rand.New(rand.NewPCG(1, 2))
	pieces := make([]string, n)
	letters := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	punct := []string{"{", "}", "[", "]", ":", ",", `"`, `":`, `",`, " ", "\n  ", ": ", "0", "12", "345"}
	for i := range pieces {
		if i < len(punct) {
			pieces[i] = punct[i]
			continue
		}
		length := 1 + rng.IntN(6)
		b := make([]byte, length)
		for j := range b {
			b[j] = letters[rng.IntN(len(letters))]
		}
		pieces[i] = string(b)
	}
	pieces[n-1] = "<eos>"
	return NewTokens(n, func(id int32) string { return pieces[id] },
		func(id int32) bool { return int(id) == n-1 })
}

const jsonish = `
root    ::= "{" ws pair (ws "," ws pair)* ws "}"
pair    ::= string ws ":" ws value
value   ::= string | number | "true" | "false" | "null"
string  ::= "\"" char* "\""
char    ::= [^"\\]
number  ::= [0-9]+
ws      ::= [ \t\n]*
`

// What one question costs, which is what the lazy check pays on nearly every
// token: the chain draws, asks about the one token it drew, and stops there.
func BenchmarkAllows(b *testing.B) {
	toks := bigVocabulary(262144)
	rules, err := Parse(jsonish)
	if err != nil {
		b.Fatal(err)
	}
	g := New(rules, toks)
	g.Accept(idOfPiece(toks, "{"))
	word := idOfPiece(toks, `"`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Allows(word)
	}
}

// What a refusal costs at its worst: the whole vocabulary tested, in the state
// that allows the least — just inside an object, where a quotation mark is
// nearly the only thing that fits.
func BenchmarkSweepTheVocabulary(b *testing.B) {
	toks := bigVocabulary(262144)
	rules, err := Parse(jsonish)
	if err != nil {
		b.Fatal(err)
	}
	g := New(rules, toks)
	g.Accept(idOfPiece(toks, "{"))

	b.ResetTimer()
	allowed := 0
	for i := 0; i < b.N; i++ {
		first := g.FirstBytes()
		allowed = 0
		for id := 0; id < toks.Size(); id++ {
			if first[g.FirstByte(int32(id))] && g.Allows(int32(id)) {
				allowed++
			}
		}
	}
	b.ReportMetric(float64(allowed), "allowed")
}

// The same sweep without the table in front of it, which is what a port of
// llama.cpp's own loop does: every token walks the stacks.
func BenchmarkSweepWithoutTheTable(b *testing.B) {
	toks := bigVocabulary(262144)
	rules, err := Parse(jsonish)
	if err != nil {
		b.Fatal(err)
	}
	g := New(rules, toks)
	g.Accept(idOfPiece(toks, "{"))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for id := 0; id < toks.Size(); id++ {
			g.Allows(int32(id))
		}
	}
}

func idOfPiece(t *Tokens, piece string) int32 {
	for id, b := range t.bytes {
		if string(b) == piece {
			return int32(id)
		}
	}
	panic(fmt.Sprintf("no token %q", piece))
}
