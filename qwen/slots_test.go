package qwen

import "testing"

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
	solo := openModel(t)
	solo.ForwardBatch(mine, 0)
	want := append([]float32(nil), solo.ForwardBatch([]int32{next}, len(mine))[0]...)

	m := openModel(t)
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
	m := openModel(t)
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
