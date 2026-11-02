package main

import (
	"fmt"
	"os"
	"testing"
)

// A Gemma checkpoint opens, with its own vocabulary: the byte-level one reads
// Gemma's file without complaint and encodes nonsense.
func TestGemmaOpens(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("GOLEM_MODEL names %s, which is not there", path)
	}
	m, err := open(path, 64, false)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	v, err := vocabulary(m)
	if err != nil {
		t.Fatal(err)
	}
	ids := v.Encode("The three laws of thermodynamics", true, false)
	if got := v.Decode(ids[1:], false); got != "The three laws of thermodynamics" {
		t.Errorf("round trip gave %q", got)
	}
	h := m.ForwardBatch(ids, 0)
	logits := make([]float32, m.Vocab())
	m.Logits(h[len(h)-1], logits)
	if next := argmax(logits); v.Decode([]int32{next}, false) == "" {
		t.Errorf("the next token %d decodes to nothing", next)
	}
}

// Every window opens with the head and carries on with the text where the last
// one stopped; with no head the text is cut as it always was.
func TestCutWindows(t *testing.T) {
	ids := []int32{10, 11, 12, 13, 14, 15, 16}
	got := cutWindows(ids, []int32{2, 7}, 5)
	want := [][]int32{{2, 7, 10, 11, 12}, {2, 7, 13, 14, 15}}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("got %v, want %v", got, want)
	}
	if got := cutWindows(ids, nil, 3); fmt.Sprint(got) != "[[10 11 12] [13 14 15]]" {
		t.Errorf("no head: got %v", got)
	}
}
