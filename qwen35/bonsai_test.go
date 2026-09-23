package qwen35

// Bonsai 2 27B on the card, against the engine that wrote it.
//
// Prism's checkpoints are ternary and rotated: every projection reads its
// activation through a sign flip and a Walsh-Hadamard transform of a thousand
// and twenty-four, and the delta net's output is reordered first. Each of those
// pieces is held on its own elsewhere; this is the check that they add up to
// the model. A rotation applied at one site too few does not crash or even
// read as noise, it answers fluently and wrongly, which is why the reference is
// the logits of Prism's own llama.cpp fork at every position and not a sample
// of text.
//
// The reference was written by the fork on the processor, from
// /mnt/data/golem-testdata/bonsai/tokens.txt, one float32 row of the vocabulary
// a position. Both packings hold the same trits, so each is held to its own
// file and both should land at the same distance.

import (
	"bufio"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"testing"
)

const bonsaiDir = "/mnt/data/LLMs_models/prism-ml/Ternary-Bonsai-2-27B-gguf/"
const bonsaiRef = "/mnt/data/golem-testdata/bonsai/"

func TestVulkanBonsaiMatchesLlamaCpp(t *testing.T) {
	for _, q := range []string{"PQ2_0", "PTQ1_0"} {
		t.Run(q, func(t *testing.T) {
			path := bonsaiDir + "Ternary-Bonsai-2-27B-" + q + ".gguf"
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no checkpoint: %v", err)
			}
			tokens := bonsaiTokens(t)
			ref, err := os.Open(bonsaiRef + "ref-" + q + ".bin")
			if err != nil {
				t.Skipf("no reference: %v", err)
			}
			defer ref.Close()

			m, err := Open(path, 1024)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if err := m.UseVulkan(); err != nil {
				t.Fatalf("vulkan: %v", err)
			}

			hs := m.ForwardBatch(tokens, 0)
			got := make([]float32, m.Cfg.Vocab)
			want := make([]float32, m.Cfg.Vocab)
			var kl float64
			agree := 0
			for i, h := range hs {
				m.Logits(h, got)
				if err := binary.Read(ref, binary.LittleEndian, want); err != nil {
					t.Fatalf("reference position %d: %v", i, err)
				}
				lp, lq := bonsaiLogSoftmax(want), bonsaiLogSoftmax(got)
				var d float64
				for j := range lp {
					d += math.Exp(lp[j]) * (lp[j] - lq[j])
				}
				kl += d
				if bonsaiArgmax(want) == bonsaiArgmax(got) {
					agree++
				}
			}
			kl /= float64(len(hs))
			t.Logf("%d positions: mean KL %.5f nats, top-1 %d/%d", len(hs), kl, agree, len(hs))
			// Measured at 0.0001 and 31 of 31 for both packings, on the card and
			// on the processor alike. A missing rotation is worth nats, not
			// hundredths of one.
			if kl > 0.02 || agree < len(hs)-2 {
				t.Errorf("mean KL %.5f, top-1 %d/%d: the card does not answer what the reference does", kl, agree, len(hs))
			}
		})
	}
}

func bonsaiTokens(t *testing.T) []int32 {
	f, err := os.Open(bonsaiRef + "tokens.txt")
	if err != nil {
		t.Skipf("no tokens: %v", err)
	}
	defer f.Close()
	var ids []int32
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		n, err := strconv.Atoi(sc.Text())
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, int32(n))
	}
	return ids
}

func bonsaiLogSoftmax(l []float32) []float64 {
	mx := float64(l[0])
	for _, v := range l {
		mx = math.Max(mx, float64(v))
	}
	var s float64
	for _, v := range l {
		s += math.Exp(float64(v) - mx)
	}
	ls := mx + math.Log(s)
	out := make([]float64, len(l))
	for i, v := range l {
		out[i] = float64(v) - ls
	}
	return out
}

func bonsaiArgmax(l []float32) int {
	bi := 0
	for i, v := range l {
		if v > l[bi] {
			bi = i
		}
	}
	return bi
}
