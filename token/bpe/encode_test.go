package bpe

// Parity against what llama.cpp recorded. The fixture is committed; a machine
// without the weights still needs them to build the Vocab, so these skip with
// the rest.

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
	path := filepath.Join(repoRoot(t), "testdata", "gemma", "tokenizer", "cases.json")
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

// Escaping is the whole of the normalization: no NFKC, no collapsing, no
// prefix. add_space_prefix is false in the file, and llama.cpp honours it.
func TestEscapeSpaces(t *testing.T) {
	cases := []struct{ in, want string }{
		{"a b", "a" + Space + "b"},
		{" ", Space},
		{"", ""},
		{"a\tb", "a\tb"},
		{"a b", "a b"}, // a non-breaking space is not a space here
	}
	for _, c := range cases {
		if got := escapeSpaces(c.in); got != c.want {
			t.Errorf("escapeSpaces(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// The pre-split is "[^\n]+|[\n]+": alternating runs, nothing dropped.
func TestSplitNewlines(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"abc", []string{"abc"}},
		{"a\nb", []string{"a", "\n", "b"}},
		{"a\n\n\nb", []string{"a", "\n\n\n", "b"}},
		{"\n\na", []string{"\n\n", "a"}},
		{"a\n", []string{"a", "\n"}},
	}
	for _, c := range cases {
		got := splitNewlines(c.in)
		if len(got) != len(c.want) {
			t.Fatalf("splitNewlines(%q) = %q, want %q", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("splitNewlines(%q) = %q, want %q", c.in, got, c.want)
			}
		}
	}
}

func TestEncodeMatchesReference(t *testing.T) {
	v := openVocab(t)

	for _, c := range loadCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			got := v.Encode(c.Text, c.AddSpecial, c.ParseSpecial)
			if !equalIDs(got, c.IDs) {
				t.Fatalf("text %q\n got %v\nwant %v", c.Text, got, c.IDs)
			}
		})
	}
}
