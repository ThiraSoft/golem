package gemma

import (
	"math"
	"os"
	"testing"
)

// A Gemma .golem checkpoint through the card against the same one through the
// processor: qwen/vulkan_golem_test.go for this engine. What Gemma adds to the
// Golem device path is its gate — ggml's tabulated GELU, where Qwen's is a
// SiLU — its four norms a block, its blocks that read another's cache, and a
// head whose logits are capped.
//
// GOLEM_MODEL_12B_GOLEM points at the file; the test says nothing without it.
func TestVulkanGolemMatchesCPU(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL_12B_GOLEM")
	if path == "" {
		t.Skip("GOLEM_MODEL_12B_GOLEM unset")
	}
	m, err := Open(path, 512)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	ids := []int32{2, 818, 5279, 529, 7001, 563, 3285, 236761, 669, 1902, 563}
	want := make([][]float32, len(ids))
	for i, h := range m.ForwardBatch(ids, 0) {
		want[i] = append([]float32(nil), h...)
	}
	blocks := make([][]float32, len(m.Cfg.Blocks))
	for i := range blocks {
		blocks[i] = append([]float32(nil), m.BlockOutput(i)...)
	}
	cpu := make([]float32, m.Cfg.Vocab)
	m.Logits(want[len(want)-1], cpu)

	m.Reset()
	if err := m.UseVulkanHead(); err != nil {
		t.Fatalf("the device head: %v", err)
	}
	dev := make([]float32, m.Cfg.Vocab)
	m.Logits(want[len(want)-1], dev)
	g := golemGap(cpu, dev)
	t.Logf("logits: worst gap %.3g, argmax %d against %d", g, Argmax(dev), Argmax(cpu))
	if g > 3e-3 {
		t.Errorf("the card's logits are %g away from the processor's", g)
	}
	if Argmax(dev) != Argmax(cpu) {
		t.Errorf("the card and the processor pick different tokens")
	}

	if err := m.UseVulkanStack(); err != nil {
		t.Fatalf("the device stack: %v", err)
	}
	m.TraceBlocks()
	m.Reset()
	device := m.ForwardBatch(ids, 0)
	for i := range blocks {
		d := golemGap(blocks[i], m.BlockOutput(i))
		if i%8 == 0 || i == len(blocks)-1 {
			t.Logf("block %2d: %.4g", i, d)
		}
		if d > 2e-2 {
			t.Errorf("block %d diverges by %g of its largest value", i, d)
			break
		}
	}
	worst := 0.0
	for i := range device {
		worst = max(worst, golemGap(want[i], device[i]))
	}
	t.Logf("%d positions, worst gap %.3g of the largest value in the hidden state", len(ids), worst)
	if worst > 2e-2 {
		t.Errorf("the card's answer is %g away from the processor's", worst)
	}
}

// golemGap is the largest difference between two vectors, as a fraction of the
// largest value in the first.
func golemGap(a, b []float32) float64 {
	var peak, worst float64
	for i := range a {
		if !math.IsInf(float64(a[i]), 0) {
			peak = max(peak, math.Abs(float64(a[i])))
		}
		if a[i] == b[i] {
			continue // both -Inf where the file forbids a token
		}
		worst = max(worst, math.Abs(float64(a[i]-b[i])))
	}
	return worst / max(peak, 1e-30)
}
