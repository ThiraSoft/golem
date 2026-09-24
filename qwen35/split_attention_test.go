package qwen35

import (
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

// TestVulkanSplitAttentionMatches holds generation through the split attention
// to the same generation through the whole one, deep into a real text.
//
// vk's TestQwenAttnSplitMatchesWhole holds the two kernels to each other on
// random numbers; this is the model around them. From one checkpoint of the
// conversation both paths take the same eight tokens one at a time, as
// generation does, and then two at once, as a speculative step does, and
// every distribution they answer is compared.
//
// GOLEM_DEPTH_MODEL names the checkpoint and GOLEM_DEPTH_TEXT the text, as for
// TestVulkanDepthProfile.
func TestVulkanSplitAttentionMatches(t *testing.T) {
	heavy.Skip(t, "it reads twelve thousand positions of a 27B")
	path := envOr("GOLEM_DEPTH_MODEL", bonsaiDir+"Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf")
	raw, err := os.ReadFile(envOr("GOLEM_DEPTH_TEXT", os.Getenv("HOME")+"/dev/golem/scratchpad/eval-wiki.txt"))
	if err != nil {
		t.Skipf("no text: %v", err)
	}
	m, err := Open(path, 16384)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	vocab, err := bytebpe.Load(m.File())
	if err != nil {
		t.Fatal(err)
	}
	tokens := vocab.Encode(string(raw), false, true)
	m.SetCheckpoints(2)
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan: %v", err)
	}
	if m.Checkpoints() != 2 {
		t.Fatalf("%d checkpoints granted, want 2", m.Checkpoints())
	}

	const steps = 8
	logits := func(h []float32) []float32 {
		out := make([]float32, m.Cfg.Vocab)
		m.Logits(h, out)
		return out
	}
	// From position at: steps tokens one at a time, then two in one pass.
	walk := func(at int, split bool) [][]float32 {
		m.gpuPipe.SetSplitAttention(split)
		if err := m.RestoreCheckpoint(1); err != nil {
			t.Fatal(err)
		}
		var out [][]float32
		for i := 0; i < steps; i++ {
			out = append(out, logits(m.ForwardBatch(tokens[at+i:at+i+1], at+i)[0]))
		}
		for _, h := range m.ForwardBatch(tokens[at+steps:at+steps+2], at+steps) {
			out = append(out, logits(h))
		}
		return out
	}
	// The engine's own noise, for scale: the first two of those tokens read
	// in one pass of two columns rather than two of one, the whole attention
	// both times. The mix goes to eight bits before its projection, so a
	// last-bit difference anywhere upstream moves a rounding.
	pair := func(at int) [][]float32 {
		m.gpuPipe.SetSplitAttention(false)
		if err := m.RestoreCheckpoint(1); err != nil {
			t.Fatal(err)
		}
		var out [][]float32
		for _, h := range m.ForwardBatch(tokens[at:at+2], at) {
			out = append(out, logits(h))
		}
		return out
	}

	filled := 0
	for _, at := range []int{1000, 6000, 12000} {
		if at+steps+2 > len(tokens) {
			t.Skipf("the text is %d tokens, %d needed", len(tokens), at+steps+2)
		}
		m.gpuPipe.SetSplitAttention(false)
		m.ForwardBatch(tokens[filled:at], filled)
		filled = at
		if err := m.SaveCheckpoint(1); err != nil {
			t.Fatal(err)
		}
		whole := walk(at, false)
		split := walk(at, true)

		worst, agree := 0.0, 0
		for i := range whole {
			kl := klDiv(whole[i], split[i])
			worst = math.Max(worst, kl)
			if bonsaiArgmax(whole[i]) == bonsaiArgmax(split[i]) {
				agree++
			}
		}
		noise := 0.0
		for i, l := range pair(at) {
			noise = math.Max(noise, klDiv(whole[i], l))
		}
		t.Logf("position %d: worst KL %.3g nat, top-1 %d/%d; one pass of two against two of one %.3g",
			at, worst, agree, len(whole), noise)
		// The engine itself is exact across pass widths (the noise above is
		// nought), so every bit of this is the split's order of addition. On
		// random numbers that order is as close to float64 as the whole
		// kernel's, closer deep in; through sixty-four blocks and the eight
		// bits the mix is rounded to it measured 4.5e-5, 1.3e-4 and 2.5e-5
		// nat at these three positions on Bonsai PQ2_0, every top-1 the same.
		if worst > 1e-3 || agree != len(whole) {
			t.Errorf("position %d: the split attention parts from the whole one (KL %.3g, top-1 %d/%d)",
				at, worst, agree, len(whole))
		}
		// The next stretch starts from the whole path's cache, which is the
		// one the checkpoint was taken on.
		if err := m.RestoreCheckpoint(1); err != nil {
			t.Fatal(err)
		}
	}
}

func klDiv(p, q []float32) float64 {
	lp, lq := bonsaiLogSoftmax(p), bonsaiLogSoftmax(q)
	kl := 0.0
	for i := range lp {
		kl += math.Exp(lp[i]) * (lp[i] - lq[i])
	}
	return kl
}
