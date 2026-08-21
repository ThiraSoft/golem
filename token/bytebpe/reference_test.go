package bytebpe

// Parity against what llama.cpp recorded. The fixture is committed; a machine
// without the weights still needs them to build the Vocab, so these skip with
// the rest.
//
// The property tests elsewhere in this package say the scanner loses nothing
// and the round trip holds. Only this file says the segmentation is the *same*
// segmentation, which is the claim that matters: a pre-tokenizer that is wrong
// by one rule still round-trips, and costs a few tokens in every sentence
// while raising nothing.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type tokenCase struct {
	Name         string  `json:"name"`
	Text         string  `json:"text"`
	AddSpecial   bool    `json:"add_special"`
	ParseSpecial bool    `json:"parse_special"`
	IDs          []int32 `json:"ids"`
	Detokenized  string  `json:"detokenized"`
}

// repoRoot walks up to the module root, so a test does not care which package
// directory it was started from.
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

func loadCases(t *testing.T) []tokenCase {
	t.Helper()
	path := filepath.Join(repoRoot(t), "testdata", "qwen", "tokenizer", "cases.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skip("the tokenizer fixtures are not on this machine")
	}
	var file struct {
		Cases []tokenCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("the fixture holds no cases")
	}
	return file.Cases
}

func equalIDs(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestEncodeMatchesReference(t *testing.T) {
	v := openVocab(t)
	for _, c := range loadCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			got := v.Encode(c.Text, c.AddSpecial, c.ParseSpecial)
			if !equalIDs(got, c.IDs) {
				t.Errorf("text %q\n got %v\nwant %v\n got %q\nwant %q",
					c.Text, got, c.IDs, pieces(v, got), pieces(v, c.IDs))
			}
		})
	}
}

func TestDecodeMatchesReference(t *testing.T) {
	v := openVocab(t)
	for _, c := range loadCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			// The recorder detokenizes with the specials rendered and nothing
			// removed, so the comparison is against the whole of what
			// llama.cpp writes back.
			if got := v.Decode(c.IDs, true); got != c.Detokenized {
				t.Errorf("Decode(%v) = %q, want %q", c.IDs, got, c.Detokenized)
			}
		})
	}
}

// pieces renders identifiers as their written forms, so a failure says which
// word was cut differently rather than which number changed.
func pieces(v *Vocab, ids []int32) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = v.Text(id)
	}
	return out
}
