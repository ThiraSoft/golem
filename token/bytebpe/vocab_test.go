package bytebpe

import (
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/merge"
)

func modelPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL_QWEN")
	if path == "" {
		t.Skip("set GOLEM_MODEL_QWEN to a Qwen3 GGUF to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("GOLEM_MODEL_QWEN names %s, which is not there", path)
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

	if v.Size() != 151936 {
		t.Fatalf("%d tokens, want 151936", v.Size())
	}
	// 151645 is <|im_end|>, which is what a Qwen turn ends on. The file
	// declares it as the end of generation; <|endoftext|> is 151643 and ends
	// one too.
	if !v.IsEOG(151645) {
		t.Error("<|im_end|> is not marked end of generation")
	}
	if v.IsEOG(1) {
		t.Error("an ordinary piece is marked end of generation")
	}
	// Unlike Gemma, this tokenizer prepends nothing.
	if v.AddBOS() {
		t.Error("the file declares add_bos_token false")
	}
}

// The merges are ranked, and the rank is the line number: a byte-alphabet
// vocabulary has no piece containing a literal space, so the separator is
// unambiguous and the first line must come back as rank 0.
func TestMergeRanks(t *testing.T) {
	v := openVocab(t)

	if _, ok := v.ranks[merge.Pair{Left: "Ġ", Right: "t"}]; !ok {
		t.Error(`the pair ("Ġ", "t") should be a merge in a GPT-2 vocabulary`)
	}
	if _, ok := v.ranks[merge.Pair{Left: "Ġ", Right: "zzzzzz"}]; ok {
		t.Error("an invented pair has a rank")
	}

	var zero int
	for _, r := range v.ranks {
		if r == 0 {
			zero++
		}
	}
	if zero != 1 {
		t.Errorf("%d pairs have rank 0, want exactly one", zero)
	}
}

// The pieces are stored in the byte alphabet, so a piece that begins a word
// begins with U+0120 rather than with a space.
func TestPiecesAreInTheByteAlphabet(t *testing.T) {
	v := openVocab(t)

	id, ok := v.ID("Ġthe")
	if !ok {
		t.Fatal(`the vocabulary has no "Ġthe", so it is not a GPT-2 one`)
	}
	if got := v.Text(id); got != "Ġthe" {
		t.Errorf("Text(%d) = %q, want %q", id, got, "Ġthe")
	}
	if _, ok := v.ID(" the"); ok {
		t.Error(`" the" with a real space should not be a piece`)
	}
}
