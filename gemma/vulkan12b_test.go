package gemma

// The 12B on the card, which is a dense checkpoint and so the other half of
// what vk/stack.go can be handed.
//
// A dense block is a mixture block with one branch: the shared branch of a
// mixture and an ordinary feed forward read the same three matrices under the
// same norm. What differs is the end of the block — one post-norm instead of
// three and no routing at all — and the tests here are the 12B's own reference
// tests run again through the device, at the tolerances they already hold to.

import (
	"testing"
	"time"
)

func load12BStack(t *testing.T) (*fixture, *Model) {
	t.Helper()
	f := loadFixture(t, "layers12")
	m := open12BEngine(t)
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	if !m.VulkanStack() {
		t.Fatal("the stack reports itself absent after being installed")
	}
	return f, m
}

// TestVulkan12BForwardBlockByBlock is TestForward12BMatchesTheReference on the
// card, at the same tolerance: a block feeds the next one, so the last number
// says nothing about where a divergence began.
func TestVulkan12BForwardBlockByBlock(t *testing.T) {
	f, m := load12BStack(t)
	m.TraceBlocks()
	for pos, token := range f.Tokens {
		m.Forward(token, pos)
		for _, il := range []int{0, 5, 24, 47} {
			compareRelative(t, "l_out-"+itoa(il)+" at position "+itoa(pos),
				m.BlockOutput(il), f.column(t, "l_out-"+itoa(il), pos), 8e-2)
		}
	}
}

// TestVulkan12BResultNorm is the last norm the logits are drawn from.
func TestVulkan12BResultNorm(t *testing.T) {
	f, m := load12BStack(t)
	var hidden []float32
	for pos, token := range f.Tokens {
		hidden = m.Forward(token, pos)
	}
	compareRelative(t, "result_norm", hidden, f.tensor(t, "result_norm"), 7e-2)
}

// TestVulkan12BGreedyMatchesTheReference replays the reference's continuation
// with every block on the card. The tie is the CPU test's, widened the way the
// mixture's was: the two paths sum the same products in a different order, and
// this prompt is a degenerate repetition whose logits sit close together.
func TestVulkan12BGreedyMatchesTheReference(t *testing.T) {
	f, m := load12BStack(t)
	if err := m.UseVulkanHead(); err != nil {
		t.Logf("the head stays on the CPU: %v", err)
	}
	pos := 0
	var hidden []float32
	for _, token := range f.Tokens {
		hidden = m.Forward(token, pos)
		pos++
	}
	const tie = 2.0
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

// The benchmarks: a whole token of the 12B, on the CPU and on the card.

func open12BEngineBench(b *testing.B) *Model {
	b.Helper()
	m, err := Open(model12BPath(b), 4096)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { m.Close() })
	return m
}

func Benchmark12BToken(b *testing.B) { benchToken(b, open12BEngineBench(b)) }

func Benchmark12BTokenVulkan(b *testing.B) {
	m := open12BEngineBench(b)
	if err := m.UseVulkanStack(); err != nil {
		b.Skipf("no Vulkan stack: %v", err)
	}
	if err := m.UseVulkanHead(); err != nil {
		b.Skipf("no Vulkan head: %v", err)
	}
	benchToken(b, m)
}

// Benchmark12BPrefillVulkan is a prompt of sixty-four positions, which the
// device path reads one column at a time.
func Benchmark12BPrefillVulkan(b *testing.B) {
	m := open12BEngineBench(b)
	if err := m.UseVulkanStack(); err != nil {
		b.Skipf("no Vulkan stack: %v", err)
	}
	if err := m.UseVulkanHead(); err != nil {
		b.Skipf("no Vulkan head: %v", err)
	}
	for _, n := range []int{64, 256} {
		b.Run(itoa(n), func(b *testing.B) {
			tokens := make([]int32, n)
			for i := range tokens {
				tokens[i] = int32(100 + i)
			}
			out := make([]float32, m.Cfg.Vocab)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.Reset()
				hidden := m.ForwardBatch(tokens, 0)
				// One head for the whole stretch, which is what llama.cpp
				// computes: a batch handed to llama_decode with a null logits
				// pointer asks for the last token's output and no other, and
				// llama-bench's test_prompt hands it the whole prompt in one
				// call. Without this the two sides are not reading the same
				// prompt — ours would be a prompt nobody drew a token from.
				m.Logits(hidden[len(hidden)-1], out)
			}
			b.StopTimer()
			b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

// TestVulkan12BPromptProfile is TestVulkanPromptProfile on the dense 12B,
// which is where the prompt sits furthest behind llama.cpp.
func TestVulkan12BPromptProfile(t *testing.T) {
	_, m := load12BStack(t)
	tl, err := m.NewStackTimeline()
	if err != nil {
		t.Fatal(err)
	}
	defer tl.Close()

	width := m.stackColumns()
	tokens := make([]int32, width)
	for i := range tokens {
		tokens[i] = int32(100 + i)
	}
	m.Reset()
	m.ForwardBatch(tokens, 0)
	m.Reset()
	m.ProfileStack(tl)
	start := time.Now()
	m.ForwardBatch(tokens, 0)
	took := time.Since(start)
	m.ProfileStack(nil)

	report, err := tl.Report(took)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d columns\n%s", width, report)
}
