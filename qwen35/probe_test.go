package qwen35

import (
	"fmt"
	"math"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

func hasNaN(v []float32) int {
	for i, f := range v {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return i
		}
	}
	return -1
}

// cpuTrunk runs the first n blocks on the CPU and returns the stream.
func (m *Model) cpuTrunk(token int32, pos, blocks int) []float32 {
	m.W.TokenEmbd.Row(int(token), m.x)
	for i := 0; i < blocks; i++ {
		Block(m.Cfg, m.Cfg.Blocks[i], &m.W.Blocks[i], &m.cache.Blocks[i], m.rope, pos, m.x, m.scratch)
	}
	return append([]float32(nil), m.x...)
}

func (m *Model) cpuMixer(token int32, pos, block int) ([]float32, []float32) {
	m.W.TokenEmbd.Row(int(token), m.x)
	for i := 0; i < block; i++ {
		Block(m.Cfg, m.Cfg.Blocks[i], &m.W.Blocks[i], &m.cache.Blocks[i], m.rope, pos, m.x, m.scratch)
	}
	bc, bw := m.Cfg.Blocks[block], &m.W.Blocks[block]
	normed := make([]float32, m.Cfg.Dim)
	copy(normed, m.x)
	nn.RMSNormPlain(normed, bw.AttnNorm, m.Cfg.Eps)
	m.scratch.SetInput(normed)
	out := make([]float32, m.Cfg.Dim)
	if bc.Type == BlockFullAttn {
		ForwardFullAttnToken(m.Cfg, bc, bw, &m.cache.Blocks[block], m.rope, pos, normed, out, m.scratch)
	} else {
		ForwardSSMToken(m.Cfg, bc, bw, &m.cache.Blocks[block], normed, out, m.scratch)
	}
	return normed, out
}

func TestVulkanProbeBlocks(t *testing.T) {
	m, err := Open(qwen38, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkanStack(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}

	tok := int32(100)
	emb := make([]float32, m.Cfg.Dim)
	m.W.TokenEmbd.Row(int(tok), emb)

	// The input norm of block 0: the same vector on both sides.
	gn, err := m.gpuPipe.ProbeNormed(emb, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	m.Reset()
	cn, _ := m.cpuMixer(tok, 0, 0)
	ma, rel := diff(cn, gn)
	fmt.Printf("block 0 input norm: max=%.4g rel=%.4g nan@%d  cpu=%.4f gpu=%.4f\n", ma, rel, hasNaN(gn), cn[:4], gn[:4])

	for _, b := range []int{0, 1, 2, 3} {
		m.Reset()
		_, cm := m.cpuMixer(tok, 0, b)
		gm, err := m.gpuPipe.ProbeMixer(emb, 0, b)
		if err != nil {
			t.Fatal(err)
		}
		ma, rel := diff(cm, gm)
		fmt.Printf("block %d mixer: max=%.4g rel=%.4g nan@%d  cpu=%.4f gpu=%.4f\n", b, ma, rel, hasNaN(gm), cm[:4], gm[:4])
	}

	for _, n := range []int{1, 2, 3, 4} {
		m.Reset()
		c := m.cpuTrunk(tok, 0, n)
		g, err := m.gpuPipe.Probe(emb, 0, n)
		if err != nil {
			t.Fatal(err)
		}
		ma, rel := diff(c, g)
		fmt.Printf("after %d blocks: max=%.4g rel=%.4g nan@%d\n", n, ma, rel, hasNaN(g))
	}
}
