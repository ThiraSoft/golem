package sentencepiece

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Pocket TTS ships one tokenizer per language, and they are not the same model:
// the French one sits under its language directory, the English one at the root
// of the snapshot. Both are checked, against a corpus of their own language.
func tokenizerPath(t *testing.T, relative string) string {
	t.Helper()
	if p := os.Getenv("POCKET_TTS_TOKENIZER"); p != "" && relative == frenchTokenizer {
		return p
	}
	pattern := filepath.Join(os.Getenv("HOME"),
		".cache/huggingface/hub/models--kyutai--pocket-tts-without-voice-cloning/snapshots/*", relative)
	found, _ := filepath.Glob(pattern)
	if len(found) == 0 {
		t.Skip("tokenizer not found — set POCKET_TTS_TOKENIZER")
	}
	return found[0]
}

const (
	frenchTokenizer  = "languages/french_24l/tokenizer.model"
	englishTokenizer = "tokenizer.model"
)

// TestParity compares the segmentation with SentencePiece's on a corpus that
// covers what a sentence runs into: elisions and contractions, accents,
// quotation marks, numbers, repeated spaces, and enough to exercise the byte
// fallback.
func TestParity(t *testing.T) {
	for _, language := range []struct{ tokenizer, cases string }{
		{frenchTokenizer, "cases.json"},
		{englishTokenizer, "cases_en.json"},
	} {
		t.Run(language.cases, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "tokenizer", language.cases))
			if err != nil {
				t.Skipf("cases missing (%v) — see ref/dump_tokens.py", err)
			}
			var reference struct {
				VocabSize int `json:"vocab_size"`
				Cases     []struct {
					Text   string   `json:"text"`
					Tokens []int    `json:"tokens"`
					Pieces []string `json:"pieces"`
				} `json:"cases"`
			}
			if err := json.Unmarshal(raw, &reference); err != nil {
				t.Fatal(err)
			}

			tok, err := Load(tokenizerPath(t, language.tokenizer))
			if err != nil {
				t.Fatal(err)
			}
			if tok.Size() != reference.VocabSize {
				t.Fatalf("vocabulary of %d pieces, want %d", tok.Size(), reference.VocabSize)
			}

			for _, c := range reference.Cases {
				got := tok.Encode(c.Text)
				if len(got) != len(c.Tokens) {
					t.Errorf("%q: %d pieces, want %d\n  got  %v\n  want %v",
						c.Text, len(got), len(c.Tokens), piecesOf(tok, got), c.Pieces)
					continue
				}
				for i := range got {
					if got[i] != c.Tokens[i] {
						t.Errorf("%q: divergence at piece %d\n  got  %v\n  want %v",
							c.Text, i, piecesOf(tok, got), c.Pieces)
						break
					}
				}
			}
		})
	}
}

func piecesOf(tok *Tokenizer, ids []int) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = tok.Piece(id)
	}
	return out
}
