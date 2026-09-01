package qwen35

import (
	"fmt"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/tensors"
)

func TestVulkanTokenCost(t *testing.T) {
	// A bench. See TestGenerationCost.
	if testing.Short() {
		t.Skip("a bench; -short is for correctness")
	}
	g, err := tensors.OpenGGUF(qwen38)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	m, err := New(g, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}

	emb := make([]float32, m.Cfg.Dim)
	m.W.TokenEmbd.Row(100, emb)

	// warm
	for i := 0; i < 3; i++ {
		if _, err := m.gpuPipe.Forward(emb, i); err != nil {
			t.Fatal(err)
		}
	}

	const n = 40
	measure := func(f func() error) time.Duration {
		t0 := time.Now()
		for i := 0; i < n; i++ {
			if err := f(); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(t0) / n
	}

	fwd := func() error { _, err := m.gpuPipe.Forward(emb, 10); return err }
	pair := [][]float32{emb, emb}
	fwd2 := func() error { _, err := m.gpuPipe.ForwardColumns(pair, []int{10, 11}); return err }

	t1 := time.Now()
	prog, err := m.gpuPipe.CompileAt(1)
	if err != nil {
		t.Fatal(err)
	}
	record := time.Since(t1)

	fullA := measure(fwd)
	replayA := measure(prog.Run)
	fullB := measure(fwd)
	replayB := measure(prog.Run)
	full := min(fullA, fullB)
	replay := min(replayA, replayB)
	prog.Close()
	fmt.Printf("full  A=%v B=%v\nreplay A=%v B=%v\n", fullA, fullB, replayA, replayB)

	logits := make([]float32, m.Cfg.Vocab)
	h := m.gpuPipe.Hidden()
	head := measure(func() error { m.Logits(h, logits); return nil })

	fmt.Printf("forward (record+submit): %v  -> %.2f t/s\n", full, 1/full.Seconds())
	fmt.Printf("recording once:          %v\n", record)
	fmt.Printf("replay only:             %v  -> %.2f t/s\n", replay, 1/replay.Seconds())
	fmt.Printf("logit head:              %v\n", head)
	fmt.Printf("token = replay + head:   %v -> %.2f t/s\n", replay+head, 1/(replay+head).Seconds())

	wide := min(measure(fwd2), measure(fwd2))
	fmt.Printf("two columns:             %v  (%.3fx one column, %.2f t/s if both land)\n",
		wide, wide.Seconds()/full.Seconds(), 2/wide.Seconds())
}
