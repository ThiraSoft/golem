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

// A wide pass against the same tokens one at a time.
//
// This is the test that was missing while the pass grew from two columns to a
// hundred and twenty-eight, and the whole of what it holds is that widening it
// changes nothing but the reading of the weights. Nothing else covered it:
// TestVulkanMatchesCPU carries four positions and TestVulkanGenerate a
// twenty-token prompt, both under the width at which the tiled product and the
// chunked mat-vec begin. Three separate faults lived under that gap at once —
// a position buffer sized for sixteen, a mat-vec binary that was never
// regenerated after its push block grew, and a thirty-two-wide form of it that
// answered a third faster and wrongly.
//
// Up to narrowChunk the demand is bit for bit, because it is the same binary
// summing the same products in the same order and any gap at all is an index.
// Above it the tiled product answers instead, which folds those products in a
// different order — a tile of the answer in registers against a running sum a
// column — so the two differ in the last places and the demand is a tolerance.
// One per cent of peak is what is measured over sixty-four blocks; gemma's
// TestVulkanBatchMatchesTokenPath says the same thing about the same trade at
// 3e-2. What must not differ is the token, which is asserted as well: an index
// fault does not drift, it answers something else.
func TestVulkanWidePassMatchesTokenPath(t *testing.T) {
	m, err := Open(qwen38, 1024)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}
	logits := make([]float32, m.Cfg.Vocab)

	// Widths on both sides of every threshold the pipeline has: the mat-vec
	// binaries, the chunk it is dispatched in, and the tiled product.
	for _, w := range []int{2, 4, 8, 16, 32, 64, 128} {
		if w > m.gpuPipe.Columns() {
			continue
		}
		xs := make([][]float32, w)
		pos := make([]int, w)
		for i := range xs {
			xs[i] = make([]float32, m.Cfg.Dim)
			m.W.TokenEmbd.Row(1000+i*7, xs[i])
			pos[i] = i
		}

		m.gpuPipe.ResetState()
		hs, err := m.gpuPipe.ForwardColumns(xs, pos)
		if err != nil {
			t.Fatal(err)
		}
		wide := append([]float32(nil), hs[w-1]...)

		m.gpuPipe.ResetState()
		var narrow []float32
		for i := range xs {
			h, err := m.gpuPipe.ForwardColumns(xs[i:i+1], pos[i:i+1])
			if err != nil {
				t.Fatal(err)
			}
			narrow = append([]float32(nil), h[0]...)
		}

		maxAbs, rel := diff(wide, narrow)
		if w <= narrowChunkTest {
			if maxAbs != 0 {
				t.Errorf("a pass of %d columns is the same binary as %d passes of one and answered differently: max|d|=%g rel=%g",
					w, w, maxAbs, rel)
			}
			continue
		}
		if rel > 0.02 {
			t.Errorf("a pass of %d columns drifted from %d passes of one by %g, past the tiled product's own folding",
				w, w, rel)
		}
		m.Logits(wide, logits)
		a := argmax(logits)
		m.Logits(narrow, logits)
		if b := argmax(logits); a != b {
			t.Errorf("a pass of %d columns named token %d where %d passes of one named %d", w, a, w, b)
		}
	}
}

// narrowChunkTest is vk's narrowChunk, which is not exported. The two have to
// agree: it is the width at or below which the same binary answers both sides
// of the comparison above.
const narrowChunkTest = 16
