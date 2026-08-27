package qwen35

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

func TestVulkanGenerate(t *testing.T) {
	g, err := tensors.OpenGGUF(qwen38)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	vocab, err := bytebpe.Load(g)
	if err != nil {
		t.Fatal(err)
	}
	m, err := New(g, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	t0 := time.Now()
	if err := m.UseVulkan(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}
	fmt.Printf("upload: %v\n", time.Since(t0).Round(time.Millisecond))

	prompt := "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n"
	toks := vocab.Encode(prompt, false, true)

	t1 := time.Now()
	hs := m.ForwardBatch(toks, 0)
	pf := time.Since(t1)
	fmt.Printf("prefill %d tok in %v (%.1f t/s)\n", len(toks), pf.Round(time.Millisecond), float64(len(toks))/pf.Seconds())

	hidden := hs[len(hs)-1]
	logits := make([]float32, m.Cfg.Vocab)
	var out strings.Builder
	pos := len(toks)
	n := 0
	t2 := time.Now()
	for ; n < 64; n++ {
		m.Logits(hidden, logits)
		best, bi := float32(-1e30), 0
		for i, v := range logits {
			if v > best {
				best, bi = v, i
			}
		}
		id := int32(bi)
		if vocab.IsEOG(id) {
			break
		}
		out.WriteString(vocab.Piece(id, false))
		hidden = m.Forward(id, pos)
		pos++
	}
	gen := time.Since(t2)
	fmt.Printf("gen %d tok in %v (%.2f t/s)\n", n, gen.Round(time.Millisecond), float64(n)/gen.Seconds())
	fmt.Printf("=== OUTPUT ===\n%s\n", out.String())
}
