package gemma

// The Vulkan head against the CPU path, on a model that has one.
//
// vk/q6k_test.go already proves the kernel against nn.MatVecQ6_K on the raw
// tensor. What is left to check here is the wiring: that the activation the
// engine hands over is the one the shader expects, and that the softcap and
// the suppressions still happen afterwards.

import (
	"math"
	"os"
	"testing"
)

// open26B is open26BEngine for a test rather than a benchmark.
func open26B(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL_26B")
	if path == "" {
		t.Skip("set GOLEM_MODEL_26B to run the Vulkan head tests")
	}
	m, err := Open(path, 4096)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestVulkanHeadMatchesCPULogits(t *testing.T) {
	m := open26B(t)

	// Something with a real hidden state behind it rather than a made-up
	// vector: a few tokens through the blocks, then the head on both paths.
	var hidden []float32
	for pos := 0; pos < 8; pos++ {
		hidden = m.Forward(int32(100+pos), pos)
	}
	state := append([]float32(nil), hidden...)

	want := make([]float32, m.Cfg.Vocab)
	m.Logits(state, want)

	if err := m.UseVulkanHead(); err != nil {
		t.Skipf("no Vulkan head: %v", err)
	}
	if !m.VulkanHead() {
		t.Fatal("the head reports itself absent after being installed")
	}
	got := make([]float32, m.Cfg.Vocab)
	m.Logits(state, got)

	var worst float64
	var where int
	for i := range want {
		if math.IsInf(float64(want[i]), -1) {
			// A suppressed token, which both paths write after the product.
			if !math.IsInf(float64(got[i]), -1) {
				t.Fatalf("token %d is suppressed on the CPU and %v on the device", i, got[i])
			}
			continue
		}
		if d := math.Abs(float64(got[i] - want[i])); d > worst {
			worst, where = d, i
		}
	}
	// The softcap is a tanh, so it compresses whatever gap the product left.
	if worst > 1e-5 {
		t.Fatalf("token %d diverges: CPU %v, device %v", where, want[where], got[where])
	}

	// And the choice itself, which is all a sampler asks of the head.
	if a, b := Argmax(want), Argmax(got); a != b {
		t.Fatalf("the two paths choose differently: %d and %d", a, b)
	}
	t.Logf("worst gap over the vocabulary: %g", worst)
}

// BenchmarkMoETokenVulkan is a whole token — the blocks on the CPU, the head
// on the card — against BenchmarkMoEToken, which is the same token with both
// on the CPU. The difference between the two is the only number that says
// whether the split was worth wiring.
func BenchmarkMoETokenVulkan(b *testing.B) {
	m := open26BEngine(b)
	if err := m.UseVulkanHead(); err != nil {
		b.Skipf("no Vulkan head: %v", err)
	}
	benchToken(b, m)
}

func BenchmarkMoEToken(b *testing.B) {
	benchToken(b, open26BEngine(b))
}

func benchToken(b *testing.B, m *Model) {
	out := make([]float32, m.Cfg.Vocab)
	for pos := 0; pos < 32; pos++ {
		m.Forward(int32(pos+100), pos)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Logits(m.Forward(1000, 32+i%64), out)
	}
}
