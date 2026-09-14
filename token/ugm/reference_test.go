package ugm

// Parity against what llama.cpp recorded, case by case: ref/nomic/corpus.tsv
// through build/ref/dump_tokens. The fixture is committed; the vocabulary is
// in the checkpoint, which is not, so the test skips without it.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

type tokenCase struct {
	Name         string  `json:"name"`
	Text         string  `json:"text"`
	AddSpecial   bool    `json:"add_special"`
	ParseSpecial bool    `json:"parse_special"`
	IDs          []int32 `json:"ids"`
}

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found above the current directory")
		}
		dir = parent
	}
}

func openVocab(t *testing.T) *Tokenizer {
	t.Helper()
	path := os.Getenv("GOLEM_MODEL_NOMIC")
	if path == "" {
		t.Skip("GOLEM_MODEL_NOMIC is not set")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })
	tok, err := Load(g)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestSegmentationMatchesLlamaCpp(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "testdata", "nomic", "tokenizer", "cases.json"))
	if err != nil {
		t.Skip("the tokenizer fixtures are not on this machine")
	}
	var file struct {
		Cases []tokenCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	tok := openVocab(t)
	for _, c := range file.Cases {
		got := tok.Encode(c.Text, c.AddSpecial, c.ParseSpecial)
		if !slices.Equal(got, c.IDs) {
			t.Errorf("%s: %q\n got  %v\n want %v", c.Name, c.Text, got, c.IDs)
		}
	}
}

// The map is what the whole package exists for, so it is pinned on its own as
// well: without it the newline would stay a newline and ① would be <unk>.
func TestTheCharacterMapFolds(t *testing.T) {
	tok := openVocab(t)
	for in, want := range map[string]string{
		"\n": " ",
		"\t": " ",
		"ﬁ":  "fi",
		"①":  "1",
		"Ａ":  "A",
		"é": "é",
		" ":  " ",
	} {
		if got, n := tok.charsmap.longest(in); n != len(in) || got != want {
			t.Errorf("%q maps to %q over %d bytes, want %q over %d", in, got, n, want, len(in))
		}
	}
}
