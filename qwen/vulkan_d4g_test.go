package qwen

import (
	"math"
	"os"
	"testing"
)

// A .golem checkpoint through the card against the same one through the
// processor, block by block.
//
// This is the acceptance test for the whole D4G device path: the two norms
// writing floats rather than Q8_0, the site vectors and the rotation on the
// activations, the four projections and the three of the feed forward, and the
// mix coming back out of its quantized form before the output projection. Any
// one of them wrong moves the hidden state, and BlockOutput says which block
// it moved in — which is the instrument this engine was built with.
//
// GOLEM_MODEL_GOLEM points at the file; the test says nothing without it.
func TestVulkanD4GMatchesCPU(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL_GOLEM")
	if path == "" {
		t.Skip("GOLEM_MODEL_GOLEM unset")
	}
	m, err := Open(path, 512)
	if err != nil {
		t.Skip(err)
	}
	defer m.Close()

	ids := []int32{9707, 11, 1879, 0, 358, 1079, 264, 1614, 315, 4128, 13}
	want := make([][]float32, len(ids))
	got := m.ForwardBatch(ids, 0)
	for i := range got {
		want[i] = append([]float32(nil), got[i]...)
	}
	blocks := make([][]float32, len(m.Cfg.Blocks))
	for i := range blocks {
		blocks[i] = append([]float32(nil), m.BlockOutput(i)...)
	}

	m.Reset()
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no device: %v", err)
	}
	m.TraceBlocks()
	m.Reset()
	device := m.ForwardBatch(ids, 0)

	// Where it first goes wrong, if it does. The device keeps a block's output
	// only when it was asked to trace, so this is silent when it was not.
	for i := range blocks {
		out := m.BlockOutput(i)
		if len(out) != len(blocks[i]) {
			break
		}
		d := worstGap(blocks[i], out)
		if i%6 == 0 || i == len(blocks)-1 {
			t.Logf("block %2d: %.4g", i, d)
		}
		if d > 3e-3 {
			t.Errorf("block %d diverges by %g of its largest value", i, d)
			break
		}
	}
	worst := 0.0
	for i := range device {
		if d := worstGap(want[i], device[i]); d > worst {
			worst = d
		}
	}
	t.Logf("%d positions, worst gap %.3g of the largest value in the hidden state", len(ids), worst)

	// And the head, which is the largest tensor in the model and the last
	// thing that was still on the processor.
	cpu := make([]float32, m.Cfg.Vocab)
	m.Logits(want[len(want)-1], cpu)
	if err := m.UseVulkanHead(); err != nil {
		t.Fatalf("the device head: %v", err)
	}
	dev := make([]float32, m.Cfg.Vocab)
	m.Logits(want[len(want)-1], dev)
	if g := worstGap(cpu, dev); g > 3e-3 {
		t.Errorf("the card's logits are %g away from the processor's", g)
	} else {
		t.Logf("logits: worst gap %.3g, argmax %d against %d", g, pick(dev), pick(cpu))
	}
	if pick(dev) != pick(cpu) {
		t.Errorf("the card and the processor pick different tokens")
	}

	// With the head on the card the stack looks the embedding up for itself,
	// which means undoing the rotation a row at a time on the device. A token
	// then crosses the bus as an identifier and the answer comes back as one
	// hidden state, and nothing else moves.
	if !m.VulkanEmbedding() {
		t.Fatal("the head is on the card and the embedding is still not")
	}
	m.Reset()
	again := m.ForwardBatch(ids, 0)
	worst = 0
	for i := range again {
		if d := worstGap(want[i], again[i]); d > worst {
			worst = d
		}
	}
	t.Logf("with the embedding on the card too: worst gap %.3g", worst)
	if worst > 3e-3 {
		t.Errorf("the device embedding moved the answer by %g", worst)
	}
	// What is left is the fp16 rounding of the queries and the scores, which
	// the two sides do in a different order. It was 2.4e-2 while the output
	// projection read the mix in its Q8_0 form.
	if worst > 3e-3 {
		t.Errorf("the card's answer is %g away from the processor's", worst)
	}
}

// pick is the token a set of logits chooses.
func pick(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// worstGap is the largest difference between two vectors, over the largest
// value either of them holds.
func worstGap(a, b []float32) float64 {
	worst, scale := 0.0, 1e-30
	for i := range a {
		if m := math.Abs(float64(a[i])); m > scale {
			scale = m
		}
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > worst {
			worst = d
		}
	}
	return worst / scale
}
