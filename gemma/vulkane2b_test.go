package gemma

// E2B on the card: the per-layer branch, the feed forward that is twice as
// wide in the second half of the model, and the blocks that read another
// block's cache, held to the reference E2B's own tests on the processor are
// held to.

import (
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

func loadE2BStack(t *testing.T) (*fixture, *Model) {
	t.Helper()
	f := loadFixture(t, "layers")
	m := openEngine(t, 4096)
	if err := m.UseVulkanStack(); err != nil {
		if strings.Contains(err.Error(), "gemma:") {
			t.Fatal(err)
		}
		t.Skipf("no Vulkan stack: %v", err)
	}
	return f, m
}

// TestVulkanE2BForwardBlockByBlock is TestForwardBlockByBlock on the card, a
// token at a time.
func TestVulkanE2BForwardBlockByBlock(t *testing.T) {
	f, m := loadE2BStack(t)
	m.TraceBlocks()
	for pos, token := range f.Tokens {
		m.Forward(token, pos)
		for _, il := range []int{0, 4, 14, 15, 19, 34} {
			compareRelative(t, "l_out-"+itoa(il)+" at position "+itoa(pos),
				m.BlockOutput(il), f.column(t, "l_out-"+itoa(il), pos), 3e-2)
		}
	}
}

// TestVulkanE2BBatchResultNorm is the prompt in one pass, which takes the
// tiled products, against the last norm the logits are drawn from.
func TestVulkanE2BBatchResultNorm(t *testing.T) {
	f, m := loadE2BStack(t)
	hidden := m.ForwardBatch(f.Tokens, 0)
	compareRelative(t, "result_norm", hidden[len(hidden)-1], f.tensor(t, "result_norm"), 2e-2)
}

// TestVulkanE2BGreedyMatchesTheReference is TestGreedyGenerationMatchesTheReference
// with the blocks and the head on the card, at the same tie.
func TestVulkanE2BGreedyMatchesTheReference(t *testing.T) {
	f, m := loadE2BStack(t)
	if err := m.UseVulkanHead(); err != nil {
		t.Logf("the head stays on the CPU: %v", err)
	}
	pos := 0
	var hidden []float32
	for _, token := range f.Tokens {
		hidden = m.Forward(token, pos)
		pos++
	}
	const tie = 0.5
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

// TestVulkanE2BWindowMasksBeyondItsReach is TestWindowMasksBeyondItsReach on
// the card, the prompt read in one pass: six hundred positions, past the
// window of 512.
func TestVulkanE2BWindowMasksBeyondItsReach(t *testing.T) {
	f := loadFixture(t, "window")
	_, m := loadE2BStack(t)
	m.TraceBlocks()
	m.ForwardBatch(f.Tokens, 0)
	for _, il := range []int{0, 4, 15} {
		compareRelative(t, "l_out-"+itoa(il)+" at the last position", m.BlockOutput(il), f.tensor(t, "l_out-"+itoa(il)), 3e-2)
	}
}

// BenchmarkE2BTokenVulkan is a token drawn with the blocks and the head on the
// card, with thirty-two positions behind it and with four thousand, where the
// eight blocks that see everything read a cache wider than their weights.
func BenchmarkE2BTokenVulkan(b *testing.B) {
	heavy.Skip(b, "a bench is a measurement, not a check, and it runs until it has one")
	m := openEngine(b, 4096)
	if err := m.UseVulkanHead(); err != nil {
		b.Skipf("no Vulkan head: %v", err)
	}
	if err := m.UseVulkanStack(); err != nil {
		b.Skipf("no Vulkan stack: %v", err)
	}
	out := make([]float32, m.Cfg.Vocab)
	for _, depth := range []int{32, 4000} {
		b.Run(itoa(depth), func(b *testing.B) {
			m.Reset()
			for pos := 0; pos < depth; pos += 256 {
				toks := make([]int32, min(256, depth-pos))
				for i := range toks {
					toks[i] = int32(100 + (pos+i)%1000)
				}
				m.ForwardBatch(toks, pos)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.Logits(m.Forward(1000, depth+i%90), out)
			}
			b.StopTimer()
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
