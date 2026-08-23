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

// Generation with a deep cache, which is where the keys and values start to
// weigh. BenchmarkForward above fills thirty-two positions and measures what
// reading the weights costs; this fills four thousand, where the cache is a
// fifth of the traffic on a dense model at full context and the format it is
// held in stops being a detail.
func BenchmarkForwardAtDepth(b *testing.B) {
	const depth = 4000
	m, err := Open(model12BPath(b), 4096)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	for pos := 0; pos < depth; pos += 256 {
		n := min(256, depth-pos)
		toks := make([]int32, n)
		for i := range toks {
			toks[i] = int32(100 + (pos+i)%1000)
		}
		m.ForwardBatch(toks, pos)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Forward(1000, depth+i%90)
	}
}
