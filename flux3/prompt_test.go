package flux3

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/krea2"
	"github.com/ThiraSoft/golem/token/bytebpe"
)

func tokenizer(t testing.TB) *krea2.Tokenizer {
	t.Helper()
	tok, err := bytebpe.LoadFiles(krea2.TokenizerDir())
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	return tok
}

func TestTokenizeMatchesComfyUI(t *testing.T) {
	f := loadFixtures(t)
	tok := tokenizer(t)

	for _, name := range []string{"p0", "p1", "p2"} {
		raw, err := os.ReadFile(filepath.Join(f.dir, "tokens", name+".json"))
		if err != nil {
			t.Fatal(err)
		}
		var want struct {
			Name        string  `json:"name"`
			Text        string  `json:"text"`
			UnpaddedLen int     `json:"unpadded_len"`
			TotalLen    int     `json:"total_len"`
			IDs         []int32 `json:"ids"`
		}
		if err := json.Unmarshal(raw, &want); err != nil {
			t.Fatal(err)
		}

		gotIDs, gotReal := Tokenize(tok, want.Text)

		if len(gotIDs)%80 != 0 {
			t.Errorf("%s: total len %d is not a multiple of 80", name, len(gotIDs))
		}
		if len(gotIDs) != want.TotalLen {
			t.Fatalf("%s: len %d, want %d", name, len(gotIDs), want.TotalLen)
		}
		if gotReal != want.UnpaddedLen {
			t.Errorf("%s: real tokens %d, want %d", name, gotReal, want.UnpaddedLen)
		}

		if len(gotIDs) != len(want.IDs) {
			t.Fatalf("%s: ids length %d != want.ids %d", name, len(gotIDs), len(want.IDs))
		}
		fixtureIDs := f.readInt32(t, "tokens/"+name+".ids")
		if len(gotIDs) != len(fixtureIDs) {
			t.Fatalf("%s: ids length %d != fixture %d", name, len(gotIDs), len(fixtureIDs))
		}
		for i := range gotIDs {
			if gotIDs[i] != want.IDs[i] {
				t.Fatalf("%s: token %d is %d, want %d", name, i, gotIDs[i], want.IDs[i])
			}
			if gotIDs[i] != fixtureIDs[i] {
				t.Fatalf("%s: token %d (%d) != fixture (%d)", name, i, gotIDs[i], fixtureIDs[i])
			}
		}
	}
}
