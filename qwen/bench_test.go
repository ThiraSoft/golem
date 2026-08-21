package qwen

import "testing"

// The two numbers the README reports, and one more that says where the first
// one goes.
//
// GOLEM_MODEL_QWEN_Q4 is what these want: four-bit weights are what anyone
// runs, and the bfloat16 file is three and a half times the size for the same
// answer.

func benchModel(b *testing.B) *Model {
	b.Helper()
	m, err := Open(quantizedPath(b), 4096)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { m.Close() })
	return m
}

// One token through the whole stack, with a prompt already in the cache — the
// steady state of generation. The logit head is measured separately because it
// alone reads the largest tensor in the file, and it is the first thing anyone
// optimizing this will want to see on its own.
func BenchmarkForward(b *testing.B) {
	m := benchModel(b)
	for pos := 0; pos < 32; pos++ {
		m.Forward(int32(pos+100), pos)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Forward(1000, 32+i%64)
	}
}

func BenchmarkLogits(b *testing.B) {
	m := benchModel(b)
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
	m := benchModel(b)
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
