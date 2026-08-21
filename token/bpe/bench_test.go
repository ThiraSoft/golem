package bpe

// What encoding costs, and where it goes.
//
// Nearly all of it is the merge loop: the escaping and the newline split are
// one pass each over the text, and the emission is one map lookup per surviving
// symbol, while the merges are a priority queue that touches every adjacent
// pair and re-queues two more for each merge it applies. That is the part worth
// a number, because it is the part anyone refactoring this will be tempted to
// make more general.

import (
	"os"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

func benchVocab(b *testing.B) *Vocab {
	b.Helper()
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		b.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this benchmark")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { g.Close() })
	v, err := Load(g)
	if err != nil {
		b.Fatal(err)
	}
	return v
}

// A paragraph of ordinary prose, which is what a prompt mostly is.
const benchProse = `The engine reads a file of weights and gives it a voice.
Nothing here is estimated: every number beside a model is a benchmark in the
repository, run on the machine named beside it. A layer is not deemed correct
until its intermediate activations match the reference implementation, and the
reference is llama.cpp itself, instrumented, because the weights on disk are
quantized and a bf16 reference would bury a mistake under its own error.`

// The same length in a shape that merges badly: long words, punctuation runs,
// and scripts whose pieces are short, so the queue stays busy for longer.
const benchAwkward = `Donaudampfschiffahrtselektrizitaetenhauptbetriebswerk...
日本語のテキストです。Привет, мир — hello 世界 🎉🇫🇷 1234567890 3.14159
func main() { fmt.Println("hi") }		a\tb\t\tc   spaced   out   text`

func BenchmarkEncodeProse(b *testing.B) {
	v := benchVocab(b)
	text := strings.Repeat(benchProse, 8)
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.Encode(text, true, false)
	}
}

func BenchmarkEncodeAwkward(b *testing.B) {
	v := benchVocab(b)
	text := strings.Repeat(benchAwkward, 8)
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.Encode(text, true, false)
	}
}

// One short line, which is what a chat turn is, and where the fixed costs of
// setting the merge up are largest relative to the work.
func BenchmarkEncodeShort(b *testing.B) {
	v := benchVocab(b)
	const text = "The capital of France is"
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.Encode(text, true, false)
	}
}
