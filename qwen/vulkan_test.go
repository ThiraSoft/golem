package qwen

// The device path against this package's own reference recordings.
//
// vk/ is proved against Gemma 4 elsewhere. What is left to show here is that
// the three differences qwen/vulkan.go names — the pre-norm block, the SiLU
// gate and the scaled scores — are the whole of the difference, and the way to
// show it is to run the quantized reference tests again through the card at
// the tolerances they already hold to.

import (
	"testing"
	"time"
)

func openQuantizedStack(t *testing.T) (*fixture, *Model) {
	t.Helper()
	f := loadFixture(t, "layers_q4")
	m := openQuantized(t)
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	if !m.VulkanStack() {
		t.Fatal("the stack reports itself absent after being installed")
	}
	return f, m
}

// TestVulkanFullStackMatchesReference is TestQuantizedFullStackMatchesReference
// on the card, block by block: a block feeds the next one, so the last number
// says nothing about where a divergence began.
//
// The tolerance is wider than that test's 3.5e-2 and the reason is the one it
// spends a page on: this path is chaotic rather than biased. Every product
// quantizes its activation to Q8_0 in blocks of thirty-two, a value sitting
// near a step rounds the other way from llama.cpp's now and then, and the two
// engines sum the same products in different orders — so the device lands on
// its own side of the same buckets. Measured, it reaches 4.4e-2 at block 27
// against the CPU path's 2.7e-2, and it grows smoothly from 7.8e-3 at block 20
// rather than jumping.
//
// What says this is drift and not a mistake is not this number. It is the two
// tests below, which do not move at all: the top sixty-four logits land within
// the same 0.5, and all sixteen greedy tokens are the ones llama.cpp drew.
func TestVulkanFullStackMatchesReference(t *testing.T) {
	f, m := openQuantizedStack(t)
	m.TraceBlocks()

	hidden := m.ForwardBatch(f.Tokens, 0)
	last := len(f.Tokens) - 1

	for i := 0; i < f.NLayer; i++ {
		name := "l_out-" + itoa(i)
		compareRelative(t, name+" at the last position", m.BlockOutput(i), f.lastColumn(t, name), 5.5e-2)
	}
	compareRelative(t, "result_norm", hidden[last], f.lastColumn(t, "result_norm"), 6e-2)
}

// And the head, which is Q4_0 here and Q6_K on the checkpoints vk/q6k.go was
// written for. The tolerance is the CPU test's.
func TestVulkanLogitsMatchReference(t *testing.T) {
	f, m := openQuantizedStack(t)
	if err := m.UseVulkanHead(); err != nil {
		t.Skipf("no Vulkan head: %v", err)
	}

	hidden := m.ForwardBatch(f.Tokens, 0)
	out := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden[len(hidden)-1], out)

	got := make([]float32, len(f.LogitsTop))
	want := make([]float32, len(f.LogitsTop))
	for i, probe := range f.LogitsTop {
		got[i], want[i] = out[probe.ID], probe.Logit
	}
	compare(t, "the top 64 logits", got, want, 0.5)

	if best := argmax(out); best != f.Argmax {
		t.Errorf("argmax %d, want %d", best, f.Argmax)
	}
}

// TestVulkanHeadMatchesCPULogits is the wiring rather than the arithmetic: the
// same hidden state through both heads, on the same model.
func TestVulkanHeadMatchesCPULogits(t *testing.T) {
	m := openQuantized(t)
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
	got := make([]float32, m.Cfg.Vocab)
	m.Logits(state, got)

	compare(t, "the whole vocabulary", got, want, 1e-3)
	if a, b := argmax(want), argmax(got); a != b {
		t.Fatalf("the two paths choose differently: %d and %d", a, b)
	}
}

// The reference's own continuation, drawn with every block and the head on the
// card.
func TestVulkanGreedyContinuationMatchesReference(t *testing.T) {
	f, m := openQuantizedStack(t)
	if err := m.UseVulkanHead(); err != nil {
		t.Skipf("no Vulkan head: %v", err)
	}

	hidden := m.ForwardBatch(f.Tokens, 0)
	out := make([]float32, m.Cfg.Vocab)
	m.Logits(hidden[len(hidden)-1], out)
	next := argmax(out)

	for step, want := range f.Greedy {
		if next != want {
			t.Fatalf("step %d: drew %d, want %d", step, next, want)
		}
		hidden = m.ForwardBatch([]int32{next}, len(f.Tokens)+step)
		m.Logits(hidden[0], out)
		next = argmax(out)
	}
}

// The benchmarks: a whole token, on the CPU and on the card.

func openQuantizedBench(b *testing.B) *Model {
	b.Helper()
	m, err := Open(quantizedPath(b), 4096)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { m.Close() })
	return m
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

func BenchmarkQuantizedToken(b *testing.B) { benchToken(b, openQuantizedBench(b)) }

func BenchmarkQuantizedTokenVulkan(b *testing.B) {
	m := openQuantizedBench(b)
	if err := m.UseVulkanHead(); err != nil {
		b.Skipf("no Vulkan head: %v", err)
	}
	if err := m.UseVulkanStack(); err != nil {
		b.Skipf("no Vulkan stack: %v", err)
	}
	benchToken(b, m)
}

// A prompt at several lengths, which is what llama-bench's -p sweeps and the
// only fair way to read the number: a pass answers a fixed width of columns
// whether or not the prompt fills it, so a prompt shorter than one pass pays
// for the columns it did not ask for.
func BenchmarkQuantizedPrefillVulkan(b *testing.B) {
	for _, n := range []int{64, 128, 256, 512} {
		b.Run(itoa(n), func(b *testing.B) {
			m := openQuantizedBench(b)
			if err := m.UseVulkanHead(); err != nil {
				b.Skipf("no Vulkan head: %v", err)
			}
			if err := m.UseVulkanStack(); err != nil {
				b.Skipf("no Vulkan stack: %v", err)
			}
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

// TestVulkanPromptProfile is not a test of anything; it is the instrument that
// says which stage of a block a stretch of prompt is spent in. Run it with -v.
func TestVulkanPromptProfile(t *testing.T) {
	_, m := openQuantizedStack(t)
	tl, err := m.NewStackTimeline()
	if err != nil {
		t.Fatal(err)
	}
	defer tl.Close()

	width := m.VulkanColumns()
	tokens := make([]int32, width)
	for i := range tokens {
		tokens[i] = int32(100 + i)
	}
	m.Reset()
	m.ForwardBatch(tokens, 0) // warm the clocks and make the recording
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
	t.Logf("%d columns in %v\n%s", width, took, report)
}
