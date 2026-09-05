package gemma

import (
	"testing"
)

// Two slots do not see each other, which is only shown by making them hold
// different conversations: a slot that continues its own after another slot
// has written a different prompt has to produce what one cache alone produces.
func TestSlotsAreIndependent(t *testing.T) {
	f := loadFixture(t, "layers")
	if len(f.Tokens) < 4 {
		t.Skipf("the fixture is %d tokens", len(f.Tokens))
	}
	mine := f.Tokens
	// Another conversation entirely: the same tokens the other way round.
	theirs := make([]int32, len(mine))
	for i, id := range mine {
		theirs[len(theirs)-1-i] = id
	}
	next := mine[0]

	// What one conversation, alone in the world, comes to.
	solo := openEngine(t, 4096)
	solo.ForwardBatch(mine, 0)
	want := append([]float32(nil), solo.ForwardBatch([]int32{next}, len(mine))[0]...)

	m := openEngine(t, 4096)
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	if m.Slots() != 2 || m.SlotContext() != 2048 {
		t.Fatalf("%d slots of %d positions, want 2 of 2048", m.Slots(), m.SlotContext())
	}

	m.UseSlot(0)
	m.ForwardBatch(mine, 0)
	m.UseSlot(1)
	m.ForwardBatch(theirs, 0)
	m.UseSlot(0)
	got := m.ForwardBatch([]int32{next}, len(mine))[0]

	if !same(want, got) {
		t.Error("slot 0 continued into what slot 1 had written")
	}
}

func TestSetSlotsRefusesAContextItCannotCut(t *testing.T) {
	m := openEngine(t, 4096)
	if err := m.SetSlots(0); err == nil {
		t.Error("zero slots was accepted")
	}
	if err := m.SetSlots(8192); err == nil {
		t.Error("more slots than positions was accepted")
	}
}

func same(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The same thing on the card, where it used to be refused rather than done.
//
// One model and one pass shape, because that is what makes the comparison bit
// for bit: a token drawn for slot 0 is one column either way, so the same
// binaries sum the same products in the same order. What changes between the
// two draws is only that another conversation has written its own prompt in
// between — which, when the ring was indexed by the position alone, was
// enough to hand slot 0 the other one's keys.
func TestSlotsAreIndependentVulkan(t *testing.T) {
	m := open26B(t)
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	mine, theirs := twoConversations(40)
	next := mine[0]

	m.UseSlot(0)
	m.ForwardBatch(mine, 0)
	alone := append([]float32(nil), m.ForwardBatch([]int32{next}, len(mine))[0]...)

	m.UseSlot(1)
	m.ForwardBatch(theirs, 0)
	m.UseSlot(0)
	after := m.ForwardBatch([]int32{next}, len(mine))[0]

	if !same(alone, after) {
		t.Error("slot 0 continued into what slot 1 had written")
	}
}

// A pass that carries a token of each of two conversations, which is what the
// server's continuous batching offers and what the card refused until the slot
// travelled beside the position.
//
// The comparison is not against the two tokens drawn one at a time. A pass of
// two columns runs a different product binary from a pass of one, and over a
// mixture a gap in the router's logits is enough to pick a different expert,
// so that comparison is worth nothing however wide the tolerance is made: the
// same conversation, no slots at all, two columns against one, comes out some
// five hundredths of the peak apart. What isolates the slot is to hold column
// 0 and its conversation still, change only what the other slot holds, and ask
// that column 0 come out unchanged — same shape, same binaries, bit for bit.
// When the ring was indexed by the position alone it did not.
func TestOneSlotOfAMixedPassIgnoresTheOther(t *testing.T) {
	m := open26B(t)
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	mine, theirs := twoConversations(40)
	other := thirdConversation(len(mine))
	last := len(mine) - 1

	m.UseSlot(0)
	m.ForwardBatch(mine[:last], 0)

	// Slot 0's token drawn beside a given neighbour. Priming slot 1 again
	// writes the same positions with other keys, which is the whole point.
	beside := func(neighbour []int32) []float32 {
		m.UseSlot(1)
		m.ForwardBatch(neighbour[:last], 0)
		got := m.ForwardMixed(
			[]int32{mine[last], neighbour[last]},
			[]Place{
				{Cache: m.Slot(0), Pos: last, Until: last},
				{Cache: m.Slot(1), Pos: last, Until: last},
			},
		)
		return append([]float32(nil), got[0]...)
	}

	want := beside(theirs)
	if got := beside(other); !same(want, got) {
		compareRelative(t, "slot 0 beside one neighbour and then another", got, want, 0)
		t.Error("what slot 0 answered depended on what slot 1 was holding")
	}
}

// The other half of it: two slots holding the same conversation are two
// columns that differ in nothing but which ring they read, so one pass has to
// answer them identically. It is the comparison that is free of the width —
// both columns are the same dispatch of the same binary.
func TestTwoSlotsHoldingOneConversationAnswerAlike(t *testing.T) {
	m := open26B(t)
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	mine, _ := twoConversations(40)
	last := len(mine) - 1

	m.UseSlot(0)
	m.ForwardBatch(mine[:last], 0)
	m.UseSlot(1)
	m.ForwardBatch(mine[:last], 0)

	got := m.ForwardMixed(
		[]int32{mine[last], mine[last]},
		[]Place{
			{Cache: m.Slot(0), Pos: last, Until: last},
			{Cache: m.Slot(1), Pos: last, Until: last},
		},
	)
	if !same(got[0], got[1]) {
		compareRelative(t, "slot 1 against slot 0 in one pass", got[1], got[0], 0)
		t.Error("two slots holding the same conversation answered differently")
	}
}

// And a whole prompt of each in one pass, which is the tiled scores kernel
// rather than the token's: a tile answers thirty-two columns off the keys of
// the union of their ranges, and two conversations cannot share that scratch
// because a position means a different entry of the cache in each of them. So
// the pass is cut into runs of one slot before the tiles are laid out. Forty
// columns a conversation is two tiles each, one of them partial.
func TestAMixedPromptKeepsATileInOneConversation(t *testing.T) {
	m := open26B(t)
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	// Uneven runs, both longer than a tile and neither a multiple of one: a
	// hundred columns is three tiles and a remainder of four, a hundred and
	// fifty-six is four and a remainder of twenty-eight.
	long, other := twoConversations(156)
	mine, theirs := long[:100], other
	third := thirdConversation(len(theirs))

	// The last state of slot 0's half of a pass that carries both prompts.
	beside := func(neighbour []int32) []float32 {
		both := append(append([]int32(nil), mine...), neighbour...)
		at := make([]Place, 0, len(both))
		for i := range mine {
			at = append(at, Place{Cache: m.Slot(0), Pos: i, Until: i})
		}
		for i := range neighbour {
			at = append(at, Place{Cache: m.Slot(1), Pos: i, Until: i})
		}
		return append([]float32(nil), m.ForwardMixed(both, at)[len(mine)-1]...)
	}

	want := beside(theirs)
	if got := beside(third); !same(want, got) {
		compareRelative(t, "slot 0's prompt beside one prompt and then another", got, want, 0)
		t.Error("slot 0's prompt came out changed by what slot 1's tiles held")
	}
}

// twoConversations is two runs of tokens with nothing in common, for the tests
// that have to tell one cache from another. The identifiers are real rows of
// the vocabulary and mean nothing beyond that.
func twoConversations(n int) (mine, theirs []int32) {
	mine, theirs = make([]int32, n), make([]int32, n)
	for i := range mine {
		mine[i] = int32(100 + i)
		theirs[i] = int32(4000 - i)
	}
	return mine, theirs
}

// thirdConversation is a run of tokens unlike either of those, for the tests
// that hold one slot still and change what the other one holds.
func thirdConversation(n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(9000 + 3*i)
	}
	return out
}

// The card's caches are laid out when the stack is built, so the slots are
// chosen before it and not after. Asking afterwards is refused rather than
// answered with a model whose rings are the wrong size.
func TestVulkanStackRefusesSlotsAskedForAfterIt(t *testing.T) {
	m := open26B(t)
	if err := m.UseVulkanStack(); err != nil {
		t.Skipf("no Vulkan stack: %v", err)
	}
	if err := m.SetSlots(2); err == nil {
		t.Error("two slots were accepted over a stack that was already built")
	}
}
