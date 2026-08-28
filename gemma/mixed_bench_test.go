package gemma

import (
	"testing"
	"time"
)

// What a pass costs as more conversations share it, and what the head costs
// beside it. One read of the weights serving n tokens is the whole claim of a
// mixed batch; this says by how much, and it is where cmd/golem-server's
// figures come from.
func TestMixedBatchCost(t *testing.T) {
	m := openEngine(t, 4096)
	const depth = 64
	for _, n := range []int{1, 2, 4, 8} {
		if err := m.SetSlots(n); err != nil {
			t.Fatal(err)
		}
		tokens := make([]int32, n)
		at := make([]Place, n)
		for i := range at {
			m.UseSlot(i)
			m.ForwardBatch(make([]int32, depth), 0)
			tokens[i] = 42
			at[i] = Place{Cache: m.Slot(i), Pos: depth}
		}
		m.ForwardMixed(tokens, at)
		start := time.Now()
		const reps = 5
		for r := 0; r < reps; r++ {
			m.ForwardMixed(tokens, at)
		}
		each := time.Since(start) / reps
		hidden := make([][]float32, n)
		out := make([][]float32, n)
		for i := range hidden {
			hidden[i] = make([]float32, m.Cfg.Dim)
			out[i] = make([]float32, m.Cfg.Vocab)
		}
		start = time.Now()
		for r := 0; r < reps; r++ {
			m.LogitsBatch(hidden, out)
		}
		head := time.Since(start) / reps
		t.Logf("%d conversations: pass %s, head %s, %.1f tokens/s",
			n, each.Round(time.Millisecond), head.Round(time.Millisecond),
			float64(n)/(each+head).Seconds())
	}
}

// Reading four prompts at once against reading one. A pass is thirty-two
// positions wide whichever way they are made up, so the question is whether
// four conversations of eight cost what one run of thirty-two costs.
func TestMixedPrefillCost(t *testing.T) {
	m := openEngine(t, 4096)
	const width = 32
	tokens := make([]int32, width)

	m.SetSlots(1)
	at := Run(m.Slot(0), 0, width)
	m.ForwardMixed(tokens, at)
	start := time.Now()
	for r := 0; r < 3; r++ {
		m.ForwardMixed(tokens, at)
	}
	one := time.Since(start) / 3

	if err := m.SetSlots(4); err != nil {
		t.Fatal(err)
	}
	four := make([]Place, 0, width)
	for i := 0; i < 4; i++ {
		four = append(four, Run(m.Slot(i), 0, width/4)...)
	}
	m.ForwardMixed(tokens, four)
	start = time.Now()
	for r := 0; r < 3; r++ {
		m.ForwardMixed(tokens, four)
	}
	split := time.Since(start) / 3

	t.Logf("32 positions of one conversation: %s (%.0f tokens/s)", one.Round(time.Millisecond), width/one.Seconds())
	t.Logf("8 positions of each of four:      %s (%.0f tokens/s)", split.Round(time.Millisecond), width/split.Seconds())
}

// The same on the card, which is where a server actually runs. One model with
// eight slots, and the widths measured off the first one, two, four and eight
// of them: the rings are laid out when the stack is built and the slots are
// chosen before it, so a second count would mean a second load of the file —
// and two of the 26B do not fit on a sixteen-gigabyte card anyway.
//
// One read of a gigabyte of weights carrying a token for each of several
// conversations is what the slot beside the position bought. Measured on an
// RX 9070 XT, pass and head both on the card, from 448 positions of context:
//
//	conversations    1      2      4      8
//	pass          11ms   16ms   16ms   22ms
//	head           1ms    2ms    5ms    9ms
//	tokens/s      80.3  110.1  195.0  256.5
//
// Which is 1.37, 2.43 and 3.19 of one conversation. What keeps it from being
// four and eight is that generation is limited by reading the weights only
// while there is nothing else to do: by eight columns the arithmetic and the
// head are what is left. The mixture takes its own cut at two — it answers one
// column by reading the eight matrices that column routed to, and two or more
// by reading the whole expert stack once instead, vk/mixture.go's byExpert —
// which is why the second conversation is worth less than the third and
// fourth. A dense checkpoint has none of that.
func TestMixedBatchCostVulkan(t *testing.T) {
	// Four hundred and forty-eight positions of context, not a handful: a
	// token drawn at position 64 costs 7.6ms on this model and one drawn at
	// 512 costs 14.1ms, because the attention reads everything before it. A
	// table taken at the shallow end would say a number no server ever sees.
	// Eight slots of a four-thousand-position context is 512 each, so this is
	// as deep as the widest case can go.
	const depth, slots = 448, 8
	m := open26B(t)
	if err := m.SetSlots(slots); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	// The head as well, because a token costs both and a server puts both
	// there: leaving the largest matrix in the model on the processor is 17ms
	// a pass against 6, and it would be the whole of what this measures.
	if err := m.UseVulkanHead(); err != nil {
		t.Skipf("no Vulkan head: %v", err)
	}
	for i := 0; i < slots; i++ {
		m.UseSlot(i)
		m.ForwardBatch(make([]int32, depth), 0)
	}
	hidden := make([][]float32, slots)
	out := make([][]float32, slots)
	for i := range hidden {
		hidden[i] = make([]float32, m.Cfg.Dim)
		out[i] = make([]float32, m.Cfg.Vocab)
	}

	for _, n := range []int{1, 2, 4, 8} {
		tokens := make([]int32, n)
		at := make([]Place, n)
		for i := range at {
			tokens[i] = 42
			at[i] = Place{Cache: m.Slot(i), Pos: depth, Until: depth}
		}
		m.ForwardMixed(tokens, at)
		start := time.Now()
		const reps = 20
		for r := 0; r < reps; r++ {
			m.ForwardMixed(tokens, at)
		}
		each := time.Since(start) / reps
		m.LogitsBatch(hidden[:n], out[:n])
		start = time.Now()
		for r := 0; r < reps; r++ {
			m.LogitsBatch(hidden[:n], out[:n])
		}
		head := time.Since(start) / reps
		t.Logf("%d conversations: pass %s, head %s, %.1f tokens/s",
			n, each.Round(time.Millisecond), head.Round(time.Millisecond),
			float64(n)/(each+head).Seconds())
	}
}

// Reading prompts on the card, one conversation against two. A pass is five
// hundred and twelve positions wide whichever way they are made up, so the
// question is whether two conversations of two hundred and fifty-six cost what
// one run of five hundred and twelve costs.
//
// They should, and the reason is the scores kernel: the pass is cut into runs
// of one conversation and each run is tiled on its own, so two conversations
// are two dispatches of the same kernel over the same total number of tiles.
// Everything else in the block — the projections, the mixture, the head — sees
// a width and not a conversation. Measured: 130ms for the one run and 88ms for
// the two, which is the shorter ranges the halves attend over — five hundred
// and twelve positions each seeing everything before them is twice the scores
// of two runs of two hundred and fifty-six.
func TestMixedPrefillCostVulkan(t *testing.T) {
	const width = 512
	m := open26B(t)
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	tokens := make([]int32, width)
	tokens[0] = 42

	time1 := func(at []Place) time.Duration {
		m.ForwardMixed(tokens, at)
		start := time.Now()
		const reps = 5
		for r := 0; r < reps; r++ {
			m.ForwardMixed(tokens, at)
		}
		return time.Since(start) / reps
	}

	one := time1(Run(m.Slot(0), 0, width))
	split := time1(append(Run(m.Slot(0), 0, width/2), Run(m.Slot(1), 0, width/2)...))
	t.Logf("%d positions of one conversation: %s (%.0f tokens/s)",
		width, one.Round(time.Millisecond), width/one.Seconds())
	t.Logf("%d positions of each of two:      %s (%.0f tokens/s)",
		width/2, split.Round(time.Millisecond), width/split.Seconds())
	t.Logf("two conversations cost %.2f of one, over the same %d positions",
		split.Seconds()/one.Seconds(), width)
}
