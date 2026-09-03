package gemma

// What the expert cache is worth on a continuation the model actually wrote.
//
// **BenchmarkMoETokenVulkan cannot answer this.** It forwards the same token
// over and over, which routes to the same eight experts of each block every
// time: any cache holding eight experts a block answers every read, and the
// figure that comes out is the cache's ceiling rather than its worth. That is
// the degeneracy trap gemma/expert_cache_test.go refuses by counting how much
// of the pool a run touches, and a benchmark cannot refuse it — so this is a
// test that prints a rate instead.
//
// It reads a prompt, then generates greedily from the model's own distribution,
// which is what a routing has to be measured over.

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/token/bpe"
)

func TestExpertCacheSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("generates a few hundred tokens on the card")
	}
	path := model26BPath(t)
	m, err := Open(path, 2048)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	// The head is not the subject and not every checkpoint has one this engine
	// can take — the 26B A4B's is Q4_0 where the shader wants Q6_K — so it is
	// taken when it is offered and the measurement stands without it. It is the
	// same on both sides of the comparison either way.
	onCard := m.UseVulkanHead() == nil
	vocab, err := bpe.Load(m.File())
	if err != nil {
		t.Skipf("no tokenizer: %v", err)
	}
	text := os.Getenv("GOLEM_CACHE_PROMPT")
	if text == "" {
		text = defaultCachePrompt
	}
	prompt := vocab.Encode(text, true, false)

	pos := 0
	var hidden []float32
	for _, tok := range prompt {
		hidden = m.Forward(tok, pos)
		pos++
	}

	// A warm-up that is not counted, because a cold cache pays for every expert
	// once and no conversation past its first few tokens does.
	logits := make([]float32, m.Cfg.Vocab)
	const warm, count = 32, 160
	for i := 0; i < warm; i++ {
		m.Logits(hidden, logits)
		hidden = m.Forward(Argmax(logits), pos)
		pos++
	}
	// The two halves are timed apart, because they are not the same question.
	// Forward is the blocks, and it is what the expert cache changes; Logits is
	// one matrix of the vocabulary, and on a checkpoint whose head this engine
	// cannot take it is a read of the whole embedding on the processor. A
	// single rate hides which of the two a run is waiting on.
	var head, blocks time.Duration
	start := time.Now()
	for i := 0; i < count; i++ {
		t0 := time.Now()
		m.Logits(hidden, logits)
		t1 := time.Now()
		hidden = m.Forward(Argmax(logits), pos)
		head += t1.Sub(t0)
		blocks += time.Since(t1)
		pos++
	}
	took := time.Since(start)
	fmt.Printf("slots=%-26s %d tokens in %v, %6.2f t/s (%6.2f ms a token, head on the card: %v)\n",
		envOrDash("GOLEM_MOE_CACHE_SLOTS"), count, took.Round(time.Millisecond),
		float64(count)/took.Seconds(), took.Seconds()*1000/float64(count), onCard)
	fmt.Printf("    of which the head %6.2f ms a token (%4.1f %%) and the blocks %6.2f ms (%4.1f %%)\n",
		head.Seconds()*1000/float64(count), 100*head.Seconds()/took.Seconds(),
		blocks.Seconds()*1000/float64(count), 100*blocks.Seconds()/took.Seconds())
}

func envOrDash(name string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	if os.Getenv("GOLEM_MOE_EXPERTS_HOST") != "" {
		return "none, all in host memory"
	}
	return "none, all resident"
}
