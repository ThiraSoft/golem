package gemma

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

func openTextModel(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this test")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Skipf("GOLEM_MODEL will not open: %v", err)
	}
	m, err := New(g, 512)
	if err != nil {
		g.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// A row given is a row used: passing the embedding of a token explicitly must
// give exactly what passing the token gives.
func TestForwardEmbeddedAgreesWithTheTable(t *testing.T) {
	m := openTextModel(t)
	tokens := []int32{2, 1596, 623, 476}

	want := m.ForwardBatch(tokens, 0)
	last := append([]float32(nil), want[len(want)-1]...)

	m.Reset()
	embeds := make([][]float32, len(tokens))
	for i := range tokens {
		row := make([]float32, m.Cfg.Dim)
		Embed(m.Cfg, m.W, tokens[i], row)
		embeds[i] = row
	}
	// The per-layer input of an embedded position is the padding token's, so a
	// batch that only wants to prove the embedding path passes token 0 for it.
	ple := make([]int32, len(tokens))
	got := m.ForwardEmbedded(tokens, embeds, ple, Run(m.cache, 0, len(tokens)))
	// With per-layer inputs in play the two are not the same computation
	// unless the identifiers agree, so this compares against the ordinary
	// path with those same identifiers.
	m.Reset()
	same := m.ForwardEmbedded(tokens, nil, ple, Run(m.cache, 0, len(tokens)))
	for i, v := range got[len(got)-1] {
		if v != same[len(same)-1][i] {
			t.Fatalf("element %d is %g through the rows, %g through the table", i, v, same[len(same)-1][i])
		}
	}
	_ = last
}

// Until widens what a position may attend to. Left at its own position — which
// is what Run writes — nothing changes, and that is the property that keeps
// the ordinary path honest.
func TestUntilDefaultsToTheOrdinaryWindow(t *testing.T) {
	m := openTextModel(t)
	tokens := []int32{2, 1596, 623, 476}

	want := m.ForwardBatch(tokens, 0)
	tail := append([]float32(nil), want[len(want)-1]...)

	m.Reset()
	got := m.ForwardMixed(tokens, Run(m.cache, 0, len(tokens)))
	for i, v := range got[len(got)-1] {
		if v != tail[i] {
			t.Fatalf("element %d moved: %g against %g", i, v, tail[i])
		}
	}
}

// And with a span, the tokens inside it see each other: the first token of the
// span must stop being what it was alone.
func TestUntilLetsASpanLookForward(t *testing.T) {
	m := openTextModel(t)
	tokens := []int32{2, 1596, 623, 476}

	causal := m.ForwardBatch(tokens, 0)
	before := append([]float32(nil), causal[1]...)

	m.Reset()
	at := Run(m.cache, 0, len(tokens))
	for i := 1; i <= 3; i++ {
		at[i].Until = 3
	}
	got := m.ForwardMixed(tokens, at)
	for i, v := range got[1] {
		if v != before[i] {
			return // it moved, which is the point
		}
	}
	t.Fatal("a token allowed to see the two after it gave the same state as one that could not")
}
