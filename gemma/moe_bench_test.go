package gemma

import (
	"os"
	"testing"
)

func open26BEngine(b *testing.B) *Model {
	b.Helper()
	path := os.Getenv("GOLEM_MODEL_26B")
	if path == "" {
		b.Skip("set GOLEM_MODEL_26B to run this benchmark")
	}
	m, err := Open(path, 4096)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { m.Close() })
	return m
}

// One token through the mixture, with a prompt already in the cache.
//
// The number to compare it against is not the 12B's: this model reads fewer
// parameters per token and more of them scattered — eight experts of a hundred
// and twenty-eight, chosen anew at every block. What it is for is watching the
// expert path move when the expert path changes.
func BenchmarkMoEForward(b *testing.B) {
	m := open26BEngine(b)
	for pos := 0; pos < 32; pos++ {
		m.Forward(int32(pos+100), pos)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Forward(1000, 32+i%64)
	}
}

// A prompt of sixty-four tokens in one pass. This is where the per-expert
// shape costs the most: sixty-four positions each read their own eight
// matrices, where the dense half reads one set for all of them.
func BenchmarkMoEPrefill(b *testing.B) {
	m := open26BEngine(b)
	tokens := make([]int32, 64)
	for i := range tokens {
		tokens[i] = int32(100 + i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Reset()
		m.ForwardBatch(tokens, 0)
	}
}
