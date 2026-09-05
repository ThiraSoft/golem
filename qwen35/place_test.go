package qwen35

import (
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// A run of text has all four components equal, which is the whole reason the
// engine has needed one integer until now.
func TestRunIsDegenerate(t *testing.T) {
	places := Run(2, 10, 3)
	if len(places) != 3 {
		t.Fatalf("%d places, wanted 3", len(places))
	}
	for i, p := range places {
		want := 10 + i
		if p.Slot != 2 {
			t.Errorf("place %d: slot %d, wanted 2", i, p.Slot)
		}
		if p.Pos != want || p.T != want || p.H != want || p.W != want {
			t.Errorf("place %d: (pos %d, t %d, h %d, w %d), wanted all %d",
				i, p.Pos, p.T, p.H, p.W, want)
		}
	}
}

// The cache index and the rotation are two numbers. An image gives several
// tokens one T and different Pos, and it is Pos the cache must count in.
func TestPlaceSeparatesCacheFromRotation(t *testing.T) {
	at := []Place{
		{Slot: 0, Pos: 5, T: 5, H: 5, W: 5},
		{Slot: 0, Pos: 6, T: 6, H: 0, W: 0},
		{Slot: 0, Pos: 7, T: 6, H: 0, W: 1},
	}
	if at[1].T != at[2].T {
		t.Fatal("the two image tokens should share one T")
	}
	if at[1].Pos == at[2].Pos {
		t.Fatal("the two image tokens must not share one cache entry")
	}
}

// The engine's own output, before and after positions grew components. Not a
// tolerance: the same float32s. Everything text does goes through the new code
// with four equal components, and if that is not exactly the old rotation then
// every fixture recorded before this change is wrong.
func TestTextIsUnmovedByPlaces(t *testing.T) {
	heavy.Skip(t, "it runs a checkpoint of tens of gigabytes")
	m, err := Open(qwen38, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()

	tokens := []int32{9707, 11, 1879, 0, 358, 1079, 264, 1273}
	byRun := m.ForwardBatch(tokens, 0)
	held := make([][]float32, len(byRun))
	for i := range byRun {
		held[i] = append([]float32(nil), byRun[i]...)
	}

	m.Reset()
	at := make([]Place, len(tokens))
	for i := range at {
		at[i] = Place{Slot: 0, Pos: i, T: i, H: i, W: i}
	}
	byPlace := m.ForwardPlaces(tokens, at)

	for c := range held {
		for i := range held[c] {
			if held[c][i] != byPlace[c][i] {
				t.Fatalf("token %d, element %d: %v against %v", c, i, held[c][i], byPlace[c][i])
			}
		}
	}
}

// The drafted column of a speculative pass sits one further on every axis. A
// draft that advanced the cache index alone would rotate by the position
// before it, which is a wrong second token and a right first one — the shape
// of bug that a rate makes look like an acceptance problem.
func TestNextAdvancesEveryAxis(t *testing.T) {
	at := Place{Slot: 1, Pos: 40, T: 12, H: 3, W: 7}
	n := at.Next()
	if n.Slot != 1 {
		t.Errorf("slot %d, wanted 1", n.Slot)
	}
	if n.Pos != 41 || n.T != 13 || n.H != 4 || n.W != 8 {
		t.Errorf("(pos %d, t %d, h %d, w %d), wanted (41, 13, 4, 8)", n.Pos, n.T, n.H, n.W)
	}
}
