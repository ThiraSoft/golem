package pockettts

import (
	"os"
	"testing"
)

// TestLanguageTable pins the four values that actually differ between the
// models Kyutai ships. They were read from the upstream YAML configs; if a
// future version moves one, this is where it should be noticed.
func TestLanguageTable(t *testing.T) {
	cases := []struct {
		name           string
		layers         int
		semicolons     bool
		pad            bool
		framesAfterEnd int
	}{
		{"french_24l", 24, true, false, 8},
		{"german", 6, true, false, 8},
		{"german_24l", 24, true, false, 8},
		{"english", 6, false, false, 8},
		{"english_2026-01", 6, false, true, 8},
		{"english_2026-04", 6, false, false, 8},
		{"spanish_24l", 24, false, false, 8},
		{"italian", 6, false, false, 8},
	}
	for _, c := range cases {
		l, err := LookupLanguage(c.name)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if l.Layers != c.layers || l.RemoveSemicolons != c.semicolons ||
			l.PadShortInputs != c.pad || l.FramesAfterEnd != c.framesAfterEnd {
			t.Errorf("%s: got %+v", c.name, l)
		}
	}
}

func TestLookupLanguage(t *testing.T) {
	if l, err := LookupLanguage(""); err != nil || l.Name != DefaultLanguage {
		t.Errorf("empty name gave %v, %v", l, err)
	}
	if _, err := LookupLanguage("klingon"); err == nil {
		t.Error("an unknown language must be refused, not guessed")
	}
	if n := len(Languages()); n != 12 {
		t.Errorf("%d languages listed, want 12", n)
	}
}

// TestTokenizerPathIrregularity guards the one layout exception upstream has.
func TestTokenizerPathIrregularity(t *testing.T) {
	odd, _ := LookupLanguage("english_2026-01")
	if got := odd.TokenizerPath(); got != "tokenizer.model" {
		t.Errorf("english_2026-01 tokenizer at %q, want the repository root", got)
	}
	normal, _ := LookupLanguage("french_24l")
	if got := normal.TokenizerPath(); got != "languages/french_24l/tokenizer.model" {
		t.Errorf("french_24l tokenizer at %q", got)
	}
}

func TestEmbeddingPath(t *testing.T) {
	lang, err := LookupLanguage("french_24l")
	if err != nil {
		t.Fatal(err)
	}
	got := lang.EmbeddingPath("alba")
	want := "languages/french_24l/embeddings/alba.safetensors"
	if got != want {
		t.Errorf("EmbeddingPath = %q, want %q", got, want)
	}
}

// Locate answers with a path or with nothing; either is a valid answer on a
// machine that may or may not have downloaded the model.
func TestLocate(t *testing.T) {
	lang, err := LookupLanguage("french_24l")
	if err != nil {
		t.Fatal(err)
	}
	if p := Locate(lang.WeightsPath()); p != "" {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("Locate returned %q, which does not exist: %v", p, err)
		}
	}
	if p := Locate("languages/french_24l/embeddings/no-such-voice.safetensors"); p != "" {
		t.Errorf("Locate found %q for a voice that does not exist", p)
	}
}
