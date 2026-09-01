package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

// A speculator whose draft is the script's next word, so accept is total, and
// one that drafts a word the model never says, so it is refused every time.
// Both count their steps: what drafting buys is passes, and a pass is what a
// test can see.
type scriptedDrafter struct {
	engine *scriptedEngine
	vocab  *wordVocab
	right  bool
	steps  int
}

// Step is qwen35.Speculator.Step's shape: the tokens decided after `token`, and
// the state of the last of them.
func (d *scriptedDrafter) Step(token int32, hidden []float32, pos int, pick func([]float32) int32) ([]int32, []float32, error) {
	d.steps++
	out := make([]float32, 4096)
	// The verifying pass draws the token after `token`, always.
	d.engine.Logits([]float32{0}, out)
	first := pick(out)
	if !d.right {
		return []int32{first}, []float32{0}, nil
	}
	// And the accepted draft draws one more from the same pass.
	d.engine.Logits([]float32{0}, out)
	return []int32{first, pick(out)}, []float32{0}, nil
}

func drafting(tb testing.TB, script []string, maxTokens int, right bool) (*Generator, *wordVocab, *scriptedDrafter, *scriptedEngine) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: script}
	for _, word := range script {
		v.id(word)
	}
	v.id("<turn|>")
	r := running(tb, e)
	d := &scriptedDrafter{engine: e, vocab: v, right: right}
	r.UseDrafter(d, func() {})
	ctx := NewContext(r, 0, 4096, time.Now, 0)
	return NewGenerator(ctx, v, wordTemplate{}, 4096, maxTokens), v, d, e
}

// The answer a drafting generator draws is the answer it draws without one.
// That is the whole contract: the prediction block decides whether a second
// token comes back for free, never what any token is.
func TestDraftingDrawsTheSameAnswer(t *testing.T) {
	script := []string{"the", "sea", "is", "wide", "<turn|>"}

	plain, _ := newGenerator(t, script, 16)
	want, err := plain.Generate(context.Background(), []int32{0}, greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, right := range []bool{true, false} {
		g, _, d, _ := drafting(t, script, 16, right)
		got, err := g.Generate(context.Background(), []int32{0}, greedy(), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got.Text != want.Text {
			t.Errorf("accepted=%v: drafting drew %q, a token at a time draws %q", right, got.Text, want.Text)
		}
		if d.steps == 0 {
			t.Errorf("accepted=%v: the generator never drafted", right)
		}
	}
}

// A draft that lands halves the passes. It is the reason any of this is here.
func TestAnAcceptedDraftHalvesThePasses(t *testing.T) {
	script := strings.Fields("one two three four five six seven eight <turn|>")

	var passes [2]int
	for i, right := range []bool{false, true} {
		g, _, d, _ := drafting(t, script, 16, right)
		if _, err := g.Generate(context.Background(), []int32{0}, greedy(), nil, nil); err != nil {
			t.Fatal(err)
		}
		passes[i] = d.steps
	}
	if passes[1]*2 > passes[0]+1 {
		t.Errorf("a draft accepted every time took %d steps against %d refused every time", passes[1], passes[0])
	}
}

// An end-of-turn marker in the second position of a draft ends the answer
// there. A drafted token that came after it must not reach the client.
func TestDraftingStopsAtTheEndOfTurn(t *testing.T) {
	g, _, _, _ := drafting(t, []string{"one", "<turn|>", "two"}, 16, true)
	got, err := g.Generate(context.Background(), []int32{0}, greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(got.Text, "two") {
		t.Errorf("the answer ran past the end of the turn: %q", got.Text)
	}
}

// Drafting is for a conversation drawing alone. Two in flight are better
// served by the pass that carries both, so the runner says no.
func TestDraftingYieldsToBatching(t *testing.T) {
	_, _, _, e := drafting(t, []string{"one"}, 4, true)
	r := running(t, e)
	r.UseDrafter(&scriptedDrafter{engine: e}, func() {})
	if !r.CanDraft() {
		t.Fatal("a lone conversation should draft")
	}
	r.Enter()
	r.Enter()
	if r.CanDraft() {
		t.Error("two conversations in flight should go through one pass, not two drafts")
	}
}
