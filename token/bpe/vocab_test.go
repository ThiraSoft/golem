package bpe

// The vocabulary is read from the file rather than written down: the E2B and
// 12B checkpoints share this tokenizer today, and nothing guarantees the next
// one will.

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/merge"
)

// modelPath is the GGUF the tests read, named by GOLEM_MODEL. A machine without
// it skips, exactly as the gemma tests do.
func modelPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("set GOLEM_MODEL to a Gemma 4 GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("GOLEM_MODEL names %s, which is not there", path)
	}
	return path
}

func openVocab(t *testing.T) *Vocab {
	t.Helper()
	g, err := tensors.OpenGGUF(modelPath(t))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	v, err := Load(g)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestLoadVocab(t *testing.T) {
	v := openVocab(t)

	if v.Size() != 262144 {
		t.Fatalf("%d tokens", v.Size())
	}
	if v.BOS() != 2 || v.EOS() != 1 {
		t.Fatalf("bos %d, eos %d", v.BOS(), v.EOS())
	}
	if !v.AddBOS() {
		t.Fatal("add_bos_token should be true")
	}

	cases := []struct {
		id   int32
		text string
		kind Kind
	}{
		{1, "<eos>", Control},
		{106, "<turn|>", Control},
		{107, "\n", Normal},
		{248, "<0x0A>", Byte},
		{506, "▁the", Normal},
	}
	for _, c := range cases {
		if got := v.Text(c.id); got != c.text {
			t.Errorf("Text(%d) = %q, want %q", c.id, got, c.text)
		}
		if got := v.Kind(c.id); got != c.kind {
			t.Errorf("Kind(%d) = %d, want %d", c.id, got, c.kind)
		}
		if id, ok := v.ID(c.text); !ok || id != c.id {
			t.Errorf("ID(%q) = %d, %v, want %d", c.text, id, ok, c.id)
		}
	}
}

// The merge separator is a real space, and both halves may themselves contain
// spaces-as-U+2581 or newlines. Splitting at index 0 would produce an empty
// left half on the entries that start with the separator's own character.
func TestMergeRanks(t *testing.T) {
	v := openVocab(t)

	if r, ok := v.ranks[merge.Pair{Left: "▁", Right: "▁"}]; !ok || r < 0 {
		t.Fatalf("the two-space merge is missing: %d, %v", r, ok)
	}
	if _, ok := v.ranks[merge.Pair{Left: "▁the", Right: "zzzzzz"}]; ok {
		t.Fatal("an invented pair has a rank")
	}
}

// The special tokens are offered longest first, because a shorter one may be a
// substring of a longer one and would then cut it in half.
func TestSpecialsAreSortedByDescendingLength(t *testing.T) {
	v := openVocab(t)

	specials := v.specialsByLength()
	if len(specials) != 24 { // 17 control + 7 user-defined; no unknown-typed token
		t.Fatalf("%d specials", len(specials))
	}
	for i := 1; i < len(specials); i++ {
		if len(v.Text(specials[i-1])) < len(v.Text(specials[i])) {
			t.Fatalf("specials are not sorted: %q before %q",
				v.Text(specials[i-1]), v.Text(specials[i]))
		}
	}
}
