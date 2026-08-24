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

// load26BVulkan is load26B with the expert stacks on the card. The tests below
// are the reference tests, run again through the other path: what has to be
// shown is not that the device agrees with this engine's CPU — it does not, to
// the bit, and cannot — but that it agrees with llama.cpp by the same margin
// the CPU does.
//
// The two sides sum the same products in different orders, and a mixture
// amplifies that: the intermediate is quantized to Q8_0 on the way into the
// second projection, so a value a hair from an integer boundary goes to the
// other side of it. Measured, that is five to thirteen magnitudes of seven
// hundred and four per expert, and a part in ten thousand on the block. The
// tolerances here are therefore the reference's own, unchanged.
func load26BVulkan(t *testing.T) (*fixture, *Model) {
	t.Helper()
	f, m := load26B(t)
	if m.Cfg.Experts == 0 {
		t.Skip("this checkpoint has no mixture blocks")
	}
	if err := m.UseVulkanExperts(); err != nil {
		t.Skipf("no Vulkan experts: %v", err)
	}
	if !m.VulkanExperts() {
		t.Fatal("the experts report themselves absent after being installed")
	}
	return f, m
}

// TestVulkanMoEForwardBlockByBlock is TestMoEForwardBlockByBlock on the card,
// at the same tolerance.
func TestVulkanMoEForwardBlockByBlock(t *testing.T) {
	f, m := load26BVulkan(t)
	for pos, token := range f.Tokens {
		m.Forward(token, pos)
		for _, il := range moeBlocks {
			compareRelative(t, "l_out-"+itoa(il)+" at position "+itoa(pos),
				m.BlockOutput(il), f.column(t, "l_out-"+itoa(il), pos), 5e-2)
		}
	}
}

// TestVulkanMoEResultNorm is the last norm the logits are drawn from.
func TestVulkanMoEResultNorm(t *testing.T) {
	f, m := load26BVulkan(t)
	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}
	compareRelative(t, "result_norm", hidden, f.tensor(t, "result_norm"), 8e-2)
}

// TestVulkanMoEGreedyMatchesTheReference replays the reference's continuation
// with the experts on the card.
//
// The tie is wider here than in the CPU test, and the reason is measured
// rather than convenient. That test's 1.5 is calibrated on the AVX2 kernel,
// which keeps eight float lanes across a row and folds them at the end; a
// kernel that sums the blocks in any other order lands somewhere else, and
// this prompt is a degenerate continuation where the model's own logits sit
// close together. At step 3 of it, this engine's portable Go path — no
// Vulkan, no AVX2, shipped and tested — chooses the same other token by 3.88,
// and the shader chooses it by 3.21. The shader is therefore nearer the
// reference than a path golem already has, and a threshold that failed it
// would be measuring the summation order rather than the device.
//
// Four is what admits both and still catches a real fault, which would not be
// three logits away but hundreds.
func TestVulkanMoEGreedyMatchesTheReference(t *testing.T) {
	f, m := load26BVulkan(t)
	pos := 0
	var hidden []float32
	for _, token := range f.Tokens {
		hidden = m.Forward(token, pos)
		pos++
	}
	const tie = 4
	logits := make([]float32, m.Cfg.Vocab)
	for step, want := range f.Greedy {
		m.Logits(hidden, logits)
		if got := Argmax(logits); got != want {
			if margin := logits[got] - logits[want]; margin > tie {
				t.Fatalf("step %d: chose %d over the reference's %d by %v, which is past a tie",
					step, got, want, margin)
			} else {
				t.Logf("step %d: chose %d over %d by %v, a tie inside the measured gap",
					step, got, want, margin)
			}
		}
		hidden = m.Forward(want, pos)
		pos++
	}
}

// BenchmarkMoETokenVulkanAll is a whole token with both the head and the
// experts on the card, against BenchmarkMoEToken on the CPU alone and
// BenchmarkMoETokenVulkan with only the head moved.
func BenchmarkMoETokenVulkanAll(b *testing.B) {
	m := open26BEngine(b)
	if err := m.UseVulkanExperts(); err != nil {
		b.Skipf("no Vulkan experts: %v", err)
	}
	if err := m.UseVulkanHead(); err != nil {
		b.Skipf("no Vulkan head: %v", err)
	}
	benchToken(b, m)
}

// BenchmarkMoETokenVulkanExperts moves the experts and leaves the head, which
// separates the two gains.
func BenchmarkMoETokenVulkanExperts(b *testing.B) {
	m := open26BEngine(b)
	if err := m.UseVulkanExperts(); err != nil {
		b.Skipf("no Vulkan experts: %v", err)
	}
	benchToken(b, m)
}

// load26BFull is load26B with everything that can go on the card: the four
// attention products, both feed-forward branches, and the logit head.
func load26BFull(t *testing.T) (*fixture, *Model) {
	t.Helper()
	f, m := load26BVulkan(t)
	if err := m.UseVulkanAttention(); err != nil {
		t.Skipf("no Vulkan attention: %v", err)
	}
	if !m.VulkanAttention() {
		t.Fatal("the attention reports itself absent after being installed")
	}
	return f, m
}

// TestVulkanFullForwardBlockByBlock is TestMoEForwardBlockByBlock with the
// attention products moved too, at the same tolerance.
func TestVulkanFullForwardBlockByBlock(t *testing.T) {
	f, m := load26BFull(t)
	for pos, token := range f.Tokens {
		m.Forward(token, pos)
		for _, il := range moeBlocks {
			compareRelative(t, "l_out-"+itoa(il)+" at position "+itoa(pos),
				m.BlockOutput(il), f.column(t, "l_out-"+itoa(il), pos), 5e-2)
		}
	}
}

// TestVulkanFullResultNorm is the last norm the logits are drawn from.
func TestVulkanFullResultNorm(t *testing.T) {
	f, m := load26BFull(t)
	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}
	compareRelative(t, "result_norm", hidden, f.tensor(t, "result_norm"), 8e-2)
}

// TestVulkanFullGreedyMatchesTheReference replays the reference's
// continuation with everything on the card. The tie is the one
// TestVulkanMoEGreedyMatchesTheReference explains and measures.
func TestVulkanFullGreedyMatchesTheReference(t *testing.T) {
	f, m := load26BFull(t)
	pos := 0
	var hidden []float32
	for _, token := range f.Tokens {
		hidden = m.Forward(token, pos)
		pos++
	}
	const tie = 4
	logits := make([]float32, m.Cfg.Vocab)
	for step, want := range f.Greedy {
		m.Logits(hidden, logits)
		if got := Argmax(logits); got != want {
			if margin := logits[got] - logits[want]; margin > tie {
				t.Fatalf("step %d: chose %d over the reference's %d by %v, which is past a tie",
					step, got, want, margin)
			} else {
				t.Logf("step %d: chose %d over %d by %v, a tie inside the measured gap",
					step, got, want, margin)
			}
		}
		hidden = m.Forward(want, pos)
		pos++
	}
}

// BenchmarkMoETokenVulkanFull is a whole token with everything that can move
// moved.
func BenchmarkMoETokenVulkanFull(b *testing.B) {
	m := open26BEngine(b)
	for _, step := range []struct {
		what string
		do   func() error
	}{
		{"experts", m.UseVulkanExperts},
		{"attention", m.UseVulkanAttention},
		{"head", m.UseVulkanHead},
	} {
		if err := step.do(); err != nil {
			b.Skipf("no Vulkan %s: %v", step.what, err)
		}
	}
	benchToken(b, m)
}
