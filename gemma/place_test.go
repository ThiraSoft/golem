package gemma

import (
	"slices"
	"testing"
)

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
	solo := openEngine(t, 4096)
	solo.ForwardBatch(mine[:len(mine)-1], 0)
	wantMine := append([]float32(nil), solo.Forward(mine[len(mine)-1], len(mine)-1)...)
	solo.Reset()
	solo.ForwardBatch(theirs[:len(theirs)-1], 0)
	wantTheirs := append([]float32(nil), solo.Forward(theirs[len(theirs)-1], len(theirs)-1)...)

	// The same two last tokens, drawn together out of two slots.
	m := openEngine(t, 4096)
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

// The head read once for several states gives each of them what it would have
// been given alone.
func TestLogitsBatchAgreesWithOneAtATime(t *testing.T) {
	f := loadFixture(t, "layers")
	m := openEngine(t, 4096)
	states := m.ForwardBatch(f.Tokens, 0)

	first := make([]float32, m.Cfg.Vocab)
	second := make([]float32, m.Cfg.Vocab)
	m.Logits(states[0], first)
	m.Logits(states[len(states)-1], second)

	together := [][]float32{make([]float32, m.Cfg.Vocab), make([]float32, m.Cfg.Vocab)}
	m.LogitsBatch([][]float32{states[0], states[len(states)-1]}, together)

	if !same(first, together[0]) || !same(second, together[1]) {
		t.Error("a state scored in a batch came out different from the same state scored alone")
	}
}

// Two continuations of one prefix written side by side in one pass, the
// second after the first in the cache, against the same two fed one after the
// other over the prefix. The second branch is turned as if it followed the
// prefix and does not see the first, so it comes out as it would alone, to the
// bit: the hole is scores of minus infinity, which add nothing to a softmax.
//
// The prefix is longer than E2B's window, so the window has to be measured
// from where a token is turned and not from where it is written.
func TestBranchesAgreeWithOneAfterAnother(t *testing.T) {
	m := openEngine(t, 4096)
	prefix := make([]int32, 600)
	for i := range prefix {
		prefix[i] = int32(100 + i*37%5000)
	}
	a, b := []int32{2000, 2001}, []int32{3000, 3001, 3002}
	n := len(prefix)

	m.ForwardBatch(prefix, 0)
	var want [][]float32
	for _, h := range m.ForwardBatch(a, n) {
		want = append(want, append([]float32(nil), h...))
	}
	for _, h := range m.ForwardBatch(b, n) {
		want = append(want, append([]float32(nil), h...))
	}

	m.Reset()
	m.ForwardBatch(prefix, 0)
	got := m.ForwardMixed(append(append([]int32{}, a...), b...), branches(m.Slot(0), n, len(a), len(b)))
	for i := range want {
		if !same(want[i], got[i]) {
			t.Errorf("position %d of the pass came out changed", i)
		}
	}
}

// branches places runs of the given lengths side by side after a prefix of n
// positions, each turned as if it followed the prefix and blind to the ones
// before it.
func branches(c *Cache, n int, lengths ...int) []Place {
	var at []Place
	from := n
	for _, l := range lengths {
		for i := range l {
			at = append(at, Place{Cache: c, Pos: from + i, Until: from + i, Shift: from - n, Holes: [2][2]int{{n, from}}})
		}
		from += l
	}
	return at
}

// An option of a second question sees the prefix and its own question's
// words, which leaves two holes: the first question before the words, and the
// option before it after them. It comes out as the same words and option fed
// over the prefix alone.
func TestTwoHolesAgreeWithOneAfterAnother(t *testing.T) {
	m := openEngine(t, 4096)
	twoHoles(t, m, func(got, want []float32) bool { return same(got, want) })
}

// twoHoles lays out, after a prefix of n: a first question's words and one
// option, then a second question's words and two options, and checks the
// last option against the second question's words and that option fed alone.
func twoHoles(t *testing.T, m *Model, agree func(got, want []float32) bool) {
	t.Helper()
	prefix := make([]int32, 600)
	for i := range prefix {
		prefix[i] = int32(100 + i*37%5000)
	}
	n := len(prefix)
	words1, opt1 := []int32{2000, 2001}, []int32{2002}
	words2, opt2a, opt2b := []int32{3000, 3001, 3002}, []int32{3003, 3004}, []int32{3005, 3006}
	prefill := func() {
		m.Reset()
		for at := 0; at < n; at += 512 {
			m.ForwardBatch(prefix[at:min(at+512, n)], at)
		}
	}

	prefill()
	alone := m.ForwardBatch(slices.Concat(words2, opt2b), n)
	want := append([]float32(nil), alone[len(alone)-1]...)

	prefill()
	c := m.Slot(0)
	var tokens []int32
	var at []Place
	put := func(ids []int32, turned int, holes [2][2]int) {
		for i, id := range ids {
			pos := n + len(tokens)
			at = append(at, Place{Cache: c, Pos: pos, Until: pos, Shift: pos - (turned + i), Holes: holes})
			tokens = append(tokens, id)
		}
	}
	put(words1, n, [2][2]int{})
	put(opt1, n+len(words1), [2][2]int{})
	w2 := n + len(tokens)
	put(words2, n, [2][2]int{{n, w2}})
	put(opt2a, n+len(words2), [2][2]int{{n, w2}})
	b := n + len(tokens)
	put(opt2b, n+len(words2), [2][2]int{{n, w2}, {w2 + len(words2), b}})
	got := m.ForwardMixed(tokens, at)
	if !agree(got[len(got)-1], want) {
		t.Error("the last option of the second question came out changed")
	}
}
