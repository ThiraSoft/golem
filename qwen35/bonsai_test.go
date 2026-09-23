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

// TestVulkanSlotsAreIndependent holds two conversations on the card and checks
// that one does not leak into the other. A delta net's state is rewritten by
// every token, so a pipeline that ran slot 1 through slot 0's state would
// continue slot 0 from the wrong recurrence and answer something else; the
// comparison is to the same continuation with nothing run in between, and it
// is exact, because the two runs are the same dispatches on the same buffers.
func TestVulkanSlotsAreIndependent(t *testing.T) {
	path := bonsaiDir + "Ternary-Bonsai-2-27B-PQ2_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	tokens := bonsaiTokens(t)
	a, next, b := tokens[:12], tokens[12], tokens[13:25]

	m, err := Open(path, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkan(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}
	if got := m.SlotContext(); got != 1024 {
		t.Fatalf("a slot holds %d positions, want 1024", got)
	}

	continued := func(interleave bool) []float32 {
		m.UseSlot(0)
		m.Reset()
		m.ForwardBatch(a, 0)
		if interleave {
			m.UseSlot(1)
			m.Reset()
			m.ForwardBatch(b, 0)
			m.UseSlot(0)
		}
		hs := m.ForwardBatch([]int32{next}, len(a))
		out := make([]float32, m.Cfg.Vocab)
		m.Logits(hs[0], out)
		return out
	}
	alone := continued(false)
	between := continued(true)
	for i := range alone {
		if alone[i] != between[i] {
			t.Fatalf("logit %d is %g with another conversation run in between and %g without", i, between[i], alone[i])
		}
	}
}

// TestVulkanSlotsShareAPass holds a pass that carries several conversations to
// the same conversations run one at a time. Each brings what golem-server
// gathers: a token drawn, or the next stretch of a prompt beside the others'
// tokens. What the pass shares is the reading of the weights; the recurrences
// and the caches are each conversation's own, and a column read against the
// wrong one answers fluently and wrongly, so the states are compared and not
// a sample.
func TestVulkanSlotsShareAPass(t *testing.T) {
	path := bonsaiDir + "Ternary-Bonsai-2-27B-PQ2_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	tokens := bonsaiTokens(t)
	prompts := [][]int32{tokens[:9], tokens[9:21], tokens[21:26]}
	// What each conversation is given next, all in one pass: a token, a
	// stretch of five, a token.
	next := [][]int32{tokens[26:27], tokens[3:8], tokens[1:2]}

	m, err := Open(path, 3*1024)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.SetSlots(3); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkan(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}
	prefill := func() {
		for s, p := range prompts {
			m.UseSlot(s)
			m.Reset()
			m.ForwardBatch(p, 0)
		}
	}

	// Each conversation on its own, a pass each.
	prefill()
	var alone [][]float32
	for s, n := range next {
		m.UseSlot(s)
		alone = append(alone, m.ForwardBatch(n, len(prompts[s]))...)
	}

	// The same, in one pass.
	prefill()
	var ids []int32
	var slots, positions []int
	for s, n := range next {
		for i, id := range n {
			ids = append(ids, id)
			slots = append(slots, s)
			positions = append(positions, len(prompts[s])+i)
		}
	}
	shared := m.ForwardSlots(ids, slots, positions)

	for c := range alone {
		worst := 0.0
		for i := range alone[c] {
			worst = math.Max(worst, math.Abs(float64(alone[c][i]-shared[c][i])))
		}
		if worst > 1e-3 {
			t.Errorf("column %d (slot %d, position %d) is %g away from the same token run alone", c, slots[c], positions[c], worst)
		}
	}

	logits := func(h []float32) []float32 {
		out := make([]float32, m.Cfg.Vocab)
		m.Logits(h, out)
		return out
	}

	// The head scores every column in one reading, and each answer is the
	// one it gives alone.
	batch := make([][]float32, len(shared))
	for c := range batch {
		batch[c] = make([]float32, m.Cfg.Vocab)
	}
	m.LogitsBatch(shared, batch)
	for c := range shared {
		one := logits(shared[c])
		for i := range one {
			if one[i] != batch[c][i] {
				t.Fatalf("column %d, logit %d: %g scored with the others and %g alone", c, i, batch[c][i], one[i])
			}
		}
	}

	// And the pass after it reads the states the shared one left, which is
	// where a state written to the wrong slot would show.
	after := make([][]float32, len(next))
	for s := range next {
		m.UseSlot(s)
		after[s] = logits(m.ForwardBatch(tokens[30:31], len(prompts[s])+len(next[s]))[0])
	}
	prefill()
	for s, n := range next {
		m.UseSlot(s)
		m.ForwardBatch(n, len(prompts[s]))
	}
	for s := range next {
		m.UseSlot(s)
		want := logits(m.ForwardBatch(tokens[30:31], len(prompts[s])+len(next[s]))[0])
		if a, b := argmax(after[s]), argmax(want); a != b {
			t.Errorf("slot %d draws %d after a shared pass and %d after its own", s, a, b)
		}
	}
}

// TestVulkanCheckpointResumes holds a prompt resumed from a checkpoint to the
// same prompt read from nothing. The copy is the delta nets' state alone, and
// the attention's cache below it is trusted to be the conversation's own, so
// a slot that went elsewhere after the copy — another prompt read past it —
// is what the test resumes, and another slot runs in between.
func TestVulkanCheckpointResumes(t *testing.T) {
	path := bonsaiDir + "Ternary-Bonsai-2-27B-PQ2_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	tokens := bonsaiTokens(t)
	shared, first, second := tokens[:14], tokens[14:22], tokens[22:30]

	m, err := Open(path, 2*1024)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	m.SetCheckpoints(2)
	if err := m.UseVulkan(); err != nil {
		t.Fatalf("vulkan: %v", err)
	}
	if m.Checkpoints() != 2 {
		t.Fatalf("%d checkpoints granted, want 2", m.Checkpoints())
	}
	last := func(hs [][]float32) []float32 {
		out := make([]float32, m.Cfg.Vocab)
		m.Logits(hs[len(hs)-1], out)
		return out
	}

	m.UseSlot(1)
	m.Reset()
	want := last(m.ForwardBatch(append(append([]int32(nil), shared...), second...), 0))

	m.Reset()
	m.ForwardBatch(shared, 0)
	if err := m.SaveCheckpoint(1); err != nil {
		t.Fatal(err)
	}
	m.ForwardBatch(first, len(shared))
	m.UseSlot(0)
	m.Reset()
	m.ForwardBatch(tokens[3:20], 0)
	m.UseSlot(1)
	if err := m.RestoreCheckpoint(1); err != nil {
		t.Fatal(err)
	}
	got := last(m.ForwardBatch(second, len(shared)))
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit %d is %g resumed from the copy and %g read from nothing", i, got[i], want[i])
		}
	}
}
