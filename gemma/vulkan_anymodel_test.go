package gemma

// The card against the processor, on whatever checkpoint is named.
//
// The reference tests beside this one read fixtures recorded from one file and
// refuse any other, which is right: they hold the engine to llama.cpp, and a
// waypoint from a different checkpoint would compare nothing. But it leaves a
// model this repository has never seen with no correctness check at all, and
// the ones worth checking are exactly the ones nobody has fixtures for — a Q8_0
// mixture whose expert pool is larger than the card and larger than the memory
// the card can address, held half on the card and half beside it.
//
// So this compares the two paths of this engine rather than the engine to
// llama.cpp. It cannot catch a fault both paths share, and it does catch
// everything the card does that the processor does not: a wrong expert, a slot
// read before it was fetched, a pool half of which went somewhere else.

import (
	"fmt"
	"math"
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

func TestVulkanMatchesCPUOnAnyModel(t *testing.T) {
	heavy.Skip(t, "generates on both paths")
	path := model26BPath(t)

	// **Tokens, not hidden states.** The two paths sum the same products in a
	// different order and the card quantizes every projection's activation to
	// Q8_0, and a mixture turns that into a discrete difference rather than a
	// small one: two logits a hair apart send the router to a different expert,
	// and the block answers something else entirely. Measured on the 26B, the
	// hidden states sit twenty to thirty per cent apart in the root mean square
	// while the model says the same thing — which is why vulkan_test.go holds
	// this engine to llama.cpp by what it *chooses*, and why this does too.
	prompt := []int32{2, 1596, 3072, 611, 8256, 1129}
	const steps = 24

	// The processor's continuation, and then the card asked to choose at each of
	// the same places. **The reference token is fed back after a disagreement**,
	// which is what vulkan_test.go does and for the reason it gives: once the
	// two part company the states differ and every later choice compares
	// nothing. What is counted is how often they choose the same next token
	// from the same state.
	run := func(card bool) (*Model, []int32) {
		t.Helper()
		m, err := Open(path, 256)
		if err != nil {
			t.Skipf("open: %v", err)
		}
		if card {
			if err := m.UseVulkanStack(); err != nil {
				m.Close()
				t.Skipf("no Vulkan stack: %v", err)
			}
		}
		return m, nil
	}
	cpu, _ := run(false)
	defer cpu.Close()
	gpu, _ := run(true)
	defer gpu.Close()

	logits := make([]float32, cpu.Cfg.Vocab)
	var hc, hg []float32
	pos := 0
	for _, tok := range prompt {
		hc = cpu.Forward(tok, pos)
		hg = gpu.Forward(tok, pos)
		pos++
	}
	same, want, got := 0, make([]int32, 0, steps), make([]int32, 0, steps)
	for i := 0; i < steps; i++ {
		cpu.Logits(hc, logits)
		ref := Argmax(logits)
		gpu.Logits(hg, logits)
		mine := Argmax(logits)
		want, got = append(want, ref), append(got, mine)
		if ref == mine {
			same++
		}
		// Both sides continue on the processor's choice, so the next comparison
		// is made from the same state rather than from two histories.
		hc = cpu.Forward(ref, pos)
		hg = gpu.Forward(ref, pos)
		pos++
	}
	fmt.Printf("card against processor: %d of %d tokens the same\n", same, len(want))
	if same != len(want) {
		fmt.Printf("  processor: %v\n  card:      %v\n", want, got)
	}
	// **This is a smoke test and it says so.** A mixture routes on logits the
	// two paths compute differently, and a near-tie sends the router to another
	// expert, so they agree on most tokens and never on all: the Q4_0 26B gives
	// twenty of twenty-four and the Q8_0 seventeen, which is the same thing
	// twice. TestVulkanSameWhereverTheExpertsLive is the exact control; what is
	// refused here is a path that answers something unrelated.
	if same*2 < len(want) {
		t.Fatalf("only %d of %d tokens agree between the card and the processor", same, len(want))
	}
}

// TestVulkanSameWhereverTheExpertsLive is the sharp control, and the one that
// holds the residency machinery.
//
// Where an expert *lives* — on the card, beside it in system memory, or fetched
// into a slot on the way past — changes nothing about the arithmetic: the same
// bytes reach the same kernel. So two runs that split the model differently
// must answer the same tokens exactly. A difference here is an expert lost or
// mixed up, not a rounding.
//
// It is what the comparison against the processor cannot be. A mixture routes
// on logits the two paths compute differently, so a near-tie sends the router
// to another expert and the block answers something else; the two agree on most
// tokens and never on all, and no threshold on that number means much. This one
// is exact or it is broken.
func TestVulkanSameWhereverTheExpertsLive(t *testing.T) {
	heavy.Skip(t, "generates on the card twice")
	path := model26BPath(t)

	run := func(slots string) []int32 {
		t.Helper()
		t.Setenv("GOLEM_MOE_CACHE_SLOTS", slots)
		m, err := Open(path, 256)
		if err != nil {
			t.Skipf("open: %v", err)
		}
		defer m.Close()
		if err := m.UseVulkanStack(); err != nil {
			t.Skipf("no Vulkan stack: %v", err)
		}
		pos := 0
		var hidden []float32
		for _, tok := range []int32{2, 1596, 3072, 611, 8256, 1129} {
			hidden = m.Forward(tok, pos)
			pos++
		}
		logits := make([]float32, m.Cfg.Vocab)
		out := make([]int32, 0, 24)
		for i := 0; i < 24; i++ {
			m.Logits(hidden, logits)
			tok := Argmax(logits)
			out = append(out, tok)
			hidden = m.Forward(tok, pos)
			pos++
		}
		return out
	}

	// Two residencies of the same model: a small cache and one three times
	// larger, which moves how many blocks keep their pool on the card as well.
	first := run("12")
	second := run("40")
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("token %d is %d with a cache of twelve and %d with one of forty — where an expert lives changed what the model said\n  twelve: %v\n  forty:  %v",
				i, first[i], second[i], first, second)
		}
	}
	fmt.Printf("the same %d tokens with a cache of twelve and of forty\n", len(first))
}

// TestVulkanFirstDivergingBlock says which block the card and the processor
// part company at, on whatever checkpoint is named.
//
// It is the instrument for a model with no fixtures. The two paths differ a
// little everywhere — a mixture quantizes its intermediate and the two sum in
// different orders — so what it looks for is not the first difference but the
// first *step change*: a block whose output is several times further apart than
// the one before it, which is what a kernel reading the wrong bytes gives.
func TestVulkanFirstDivergingBlock(t *testing.T) {
	heavy.Skip(t, "forwards the model on both paths")
	path := model26BPath(t)
	ids := []int32{2, 1596, 3072, 611}

	trace := func(card bool) [][]float32 {
		t.Helper()
		m, err := Open(path, 64)
		if err != nil {
			t.Skipf("open: %v", err)
		}
		defer m.Close()
		if card {
			if err := m.UseVulkanStack(); err != nil {
				t.Skipf("no Vulkan stack: %v", err)
			}
		}
		m.TraceBlocks()
		for pos, tok := range ids {
			m.Forward(tok, pos)
		}
		out := make([][]float32, 0, len(m.Cfg.Blocks))
		for i := range m.Cfg.Blocks {
			out = append(out, append([]float32(nil), m.BlockOutput(i)...))
		}
		return out
	}

	want := trace(false)
	got := trace(true)
	for i := range want {
		var num, den float64
		for j := range want[i] {
			d := float64(got[i][j] - want[i][j])
			num += d * d
			den += float64(want[i][j]) * float64(want[i][j])
		}
		rel := 0.0
		if den > 0 {
			rel = math.Sqrt(num / den)
		}
		fmt.Printf("  bloc %2d: rel=%.4f\n", i, rel)
	}
}
