package gemma

// What the feed-forward half costs on the card, apart from the token around it.

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// BenchmarkMixtureBlock is one block's whole feed-forward half on the card,
// submission and all. Multiply it by the block count for what a token spends
// there.
func BenchmarkMixtureBlock(b *testing.B) {
	path := os.Getenv("GOLEM_MODEL_26B")
	if path == "" {
		b.Skip("no model")
	}
	m, err := Open(path, 512)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	if err := m.UseVulkanExperts(); err != nil {
		b.Skip(err)
	}
	cfg, s := m.Cfg, m.scratch
	bw := &m.W.Blocks[0]
	m.Forward(100, 0)

	in := nn.NewBatch(cfg.Dim, 1)
	copy(in.F[0], s.resid[0])
	in.QuantizeColumnRange(0, 0, cfg.Dim)
	ids := make([]int32, cfg.ExpertsUsed)
	cw := make([]float32, cfg.ExpertsUsed)
	for k := range ids {
		ids[k] = int32(k * 7)
		cw[k] = 0.125
	}
	so := make([]float32, cfg.Dim)
	eo := make([]float32, cfg.Dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := bw.Mixture.Run(bw.MixtureIndex, in, in, ids, cw, so, eo); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkMixtureSaturated is the same work with thirty-two blocks to a
// submission, which is what the kernels cost when the card is awake. The gap
// between this and BenchmarkMixtureBlock is not the shader: it is the round
// trip, and the clocks the round trip never raises.
func BenchmarkMixtureSaturated(b *testing.B) {
	path := os.Getenv("GOLEM_MODEL_26B")
	if path == "" {
		b.Skip("no model")
	}
	m, err := Open(path, 512)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	if err := m.UseVulkanExperts(); err != nil {
		b.Skip(err)
	}
	cfg, s := m.Cfg, m.scratch
	bw := &m.W.Blocks[0]
	m.Forward(100, 0)
	in := nn.NewBatch(cfg.Dim, 1)
	copy(in.F[0], s.resid[0])
	in.QuantizeColumnRange(0, 0, cfg.Dim)
	ids := make([]int32, cfg.ExpertsUsed)
	cw := make([]float32, cfg.ExpertsUsed)
	for k := range ids {
		ids[k] = int32(k * 7)
		cw[k] = 0.125
	}
	so, eo := make([]float32, cfg.Dim), make([]float32, cfg.Dim)
	if err := bw.Mixture.Run(bw.MixtureIndex, in, in, ids, cw, so, eo); err != nil {
		b.Fatal(err)
	}
	const rounds = 32
	// experts: 8 x 2 x 704 rows of 2816, plus down; shared: 2 x 2112 and down.
	perBlock := int64(8*2*cfg.ExpertFFN*cfg.Dim/32*18 + 8*cfg.Dim*cfg.ExpertFFN/32*18 +
		2*cfg.Blocks[0].FFN*cfg.Dim/32*18 + cfg.Dim*cfg.Blocks[0].FFN/32*18)
	b.SetBytes(perBlock * rounds)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := m.experts.RunTimes(bw.MixtureIndex, rounds); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAttentionBlock is one block's whole attention on the card,
// submission and all: the four products, the norms, the rotation, the cache,
// the scores and the mix.
func BenchmarkAttentionBlock(b *testing.B) {
	path := os.Getenv("GOLEM_MODEL_26B")
	if path == "" {
		b.Skip("no model")
	}
	m, err := Open(path, 512)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	if err := m.UseVulkanAttention(); err != nil {
		b.Skip(err)
	}
	cfg := m.Cfg
	bw, bc := &m.W.Blocks[0], cfg.Blocks[0]
	m.Forward(100, 0)

	in := nn.NewBatch(cfg.Dim, 1)
	for i := range in.F[0] {
		in.F[0][i] = float32(i%64) * 0.01
	}
	in.QuantizeColumnRange(0, 0, cfg.Dim)
	cos := make([]float32, bc.RoPEDims/2)
	sin := make([]float32, bc.RoPEDims/2)
	for i := range cos {
		cos[i], sin[i] = 1, 0
	}
	out := make([]float32, cfg.Dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := bw.Attn.Attend(bw.AttnIndex, in, cos, sin, 1, 0, 1, out); err != nil {
			b.Fatal(err)
		}
	}
}
