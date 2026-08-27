package qwen35

import (
	"fmt"
	"math"
	"testing"
	"time"
)

const qwen38 = "/mnt/data/LLMs_models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf"

func diff(a, b []float32) (maxAbs, rel float64) {
	var num, den float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxAbs {
			maxAbs = d
		}
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	return maxAbs, math.Sqrt(num / (den + 1e-30))
}

func TestVulkanMatchesCPU(t *testing.T) {
	m, err := Open(qwen38, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()

	toks := []int32{100, 200, 300, 400}
	cpu := make([][]float32, len(toks))
	cpuTop := make([]int32, len(toks))
	logits := make([]float32, m.Cfg.Vocab)
	for i, tok := range toks {
		cpu[i] = append([]float32(nil), m.Forward(tok, i)...)
		m.Logits(cpu[i], logits)
		cpuTop[i] = argmax(logits)
	}

	t0 := time.Now()
	if err := m.UseVulkanStack(); err != nil {
		t.Fatalf("vulkan stack: %v", err)
	}
	fmt.Printf("uploaded in %v\n", time.Since(t0).Round(time.Millisecond))
	m.Reset()

	// The two paths do not agree to the last bit and are not meant to. The
	// processor quantises the activation of every output projection to Q8_0,
	// which is what llama.cpp does and what the Q4_0 kernels there read; the
	// card feeds those same products the floats. Over sixty-four blocks that
	// compounds to a few per cent, in the direction of more precision rather
	// than less — so what is asserted is the thing that has to hold, which is
	// that both paths name the same token.
	for i, tok := range toks {
		g := m.Forward(tok, i)
		maxAbs, rel := diff(cpu[i], g)
		m.Logits(g, logits)
		top := argmax(logits)
		fmt.Printf("pos %d: max|d|=%.4g  rel=%.4g  token cpu=%d gpu=%d\n", i, maxAbs, rel, cpuTop[i], top)
		if hasNaN(g) >= 0 {
			t.Fatalf("position %d has a hidden state that is not a number", i)
		}
		if rel > 0.1 {
			t.Errorf("position %d diverges far past the quantisation gap: relative error %.4g", i, rel)
		}
		if top != cpuTop[i] {
			t.Errorf("position %d: the card draws %d where the processor draws %d", i, top, cpuTop[i])
		}
		_ = tok
	}
}

// TestVulkanTwoColumns checks that a pass carrying two tokens answers what two
// passes of one answer. The delta net's state and the attention's cache both
// run from one column to the next inside the pass, which is the part that a
// per-column dispatch cannot do for itself.
func TestVulkanTwoColumns(t *testing.T) {
	m, err := Open(qwen38, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkanStack(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}

	toks := []int32{100, 7891, 402, 55}
	emb := make([][]float32, len(toks))
	for i, tok := range toks {
		emb[i] = make([]float32, m.Cfg.Dim)
		m.W.TokenEmbd.Row(int(tok), emb[i])
	}

	// One at a time, from a clean state.
	m.Reset()
	one := make([][]float32, len(toks))
	for i := range toks {
		h, err := m.gpuPipe.Forward(emb[i], i)
		if err != nil {
			t.Fatal(err)
		}
		one[i] = append([]float32(nil), h...)
	}

	// The same tokens, two columns to a pass.
	m.Reset()
	two := make([][]float32, len(toks))
	for i := 0; i < len(toks); i += 2 {
		out, err := m.gpuPipe.ForwardColumns(emb[i:i+2], []int{i, i + 1})
		if err != nil {
			t.Fatal(err)
		}
		two[i] = append([]float32(nil), out[0]...)
		two[i+1] = append([]float32(nil), out[1]...)
	}

	for i := range toks {
		maxAbs, rel := diff(one[i], two[i])
		fmt.Printf("column %d: max|d|=%.4g rel=%.4g\n", i, maxAbs, rel)
		if hasNaN(two[i]) >= 0 {
			t.Fatalf("column %d is not a number", i)
		}
		if rel > 1e-4 {
			t.Errorf("column %d differs from the one-column pass: rel=%.4g", i, rel)
		}
	}
}
