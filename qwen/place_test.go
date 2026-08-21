package qwen

import "testing"

// A batch drawn from two conversations gives each token exactly what it would
// have been given alone. Bit for bit: the weights are read once for the batch,
// and nothing about a token's arithmetic depends on what it is batched with.
func TestMixedBatchAgreesWithOneAtATime(t *testing.T) {
	f := loadFixture(t, "layers")
	if len(f.Tokens) < 4 {
		t.Skipf("the fixture is %d tokens", len(f.Tokens))
	}
	mine := f.Tokens
	theirs := make([]int32, len(mine))
	for i, id := range mine {
		theirs[len(theirs)-1-i] = id
	}

	// What each conversation comes to on its own, one cache at a time.
	solo := openModel(t)
	solo.ForwardBatch(mine[:len(mine)-1], 0)
	wantMine := append([]float32(nil), solo.Forward(mine[len(mine)-1], len(mine)-1)...)
	solo.Reset()
	solo.ForwardBatch(theirs[:len(theirs)-1], 0)
	wantTheirs := append([]float32(nil), solo.Forward(theirs[len(theirs)-1], len(theirs)-1)...)

	// The same two last tokens, drawn together out of two slots.
	m := openModel(t)
	if err := m.SetSlots(2); err != nil {
		t.Fatal(err)
	}
	m.UseSlot(0)
	m.ForwardBatch(mine[:len(mine)-1], 0)
	m.UseSlot(1)
	m.ForwardBatch(theirs[:len(theirs)-1], 0)

	last := len(mine) - 1
	got := m.ForwardMixed(
		[]int32{mine[last], theirs[last]},
		[]Place{{Cache: m.Slot(0), Pos: last}, {Cache: m.Slot(1), Pos: last}},
	)
	if !same(wantMine, got[0]) {
		t.Error("the first conversation's token came out of the mixed batch changed")
	}
	if !same(wantTheirs, got[1]) {
		t.Error("the second conversation's token came out of the mixed batch changed")
	}
}
