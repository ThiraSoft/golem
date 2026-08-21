package bpe

import (
	"testing"

	"github.com/ThiraSoft/golem/token/special"
)

// A special token cuts the text in three, and the halves are rescanned by the
// shorter candidates. Taking the longest first is what keeps "<|turn>" from
// being eaten by a shorter neighbour that happens to be a substring.
func TestPartition(t *testing.T) {
	v := openVocab(t)

	turn, ok := v.ID("<turn|>")
	if !ok {
		t.Fatal("<turn|> is missing from the vocabulary")
	}

	got := special.Partition("before <turn|> after", v.specials, true)
	want := []special.Fragment{
		{Text: "before ", ID: -1},
		{ID: turn},
		{Text: " after", ID: -1},
	}
	if len(got) != len(want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("got %+v, want %+v", got, want)
		}
	}
}

// With parse_special off, a control token is five ordinary characters again.
func TestPartitionLeavesControlTokensAloneWhenNotParsing(t *testing.T) {
	v := openVocab(t)

	got := special.Partition("before <turn|> after", v.specials, false)
	if len(got) != 1 || got[0].ID != -1 || got[0].Text != "before <turn|> after" {
		t.Fatalf("got %+v", got)
	}
}

// Two specials back to back leave no raw fragment between them.
func TestPartitionAdjacentSpecials(t *testing.T) {
	v := openVocab(t)

	bos, _ := v.ID("<bos>")
	turn, _ := v.ID("<turn|>")

	got := special.Partition("<bos><turn|>x", v.specials, true)
	if len(got) != 3 {
		t.Fatalf("got %+v", got)
	}
	if got[0].ID != bos || got[1].ID != turn || got[2].Text != "x" {
		t.Fatalf("got %+v", got)
	}
}
