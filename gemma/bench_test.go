package gemma

import "testing"

// One token through the whole stack, with a prompt already in the cache — the
// steady state of generation. The logit head is measured separately because it
// alone reads three quarters of a gigabyte, and it is the first thing anyone
// optimizing this will want to see on its own.
func BenchmarkForward(b *testing.B) {
	m := openEngine(b, 4096)
	for pos := 0; pos < 32; pos++ {
		m.Forward(int32(pos+100), pos)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Forward(1000, 32+i%64)
	}
}

func BenchmarkLogits(b *testing.B) {
	m := openEngine(b, 4096)
	hidden := m.Forward(2, 0)
	out := make([]float32, m.Cfg.Vocab)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Logits(hidden, out)
	}
}

// A prompt of sixty-four tokens, read in one pass. This is the measurement the
// batch exists for: the same weights, read once instead of sixty-four times.
func BenchmarkPrefill(b *testing.B) {
	m := openEngine(b, 4096)
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
