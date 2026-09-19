package krea2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/token/bytebpe"
)

func vocab(t testing.TB) *bytebpe.Vocab {
	t.Helper()
	v, err := bytebpe.LoadFiles(TokenizerDir())
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	return v
}

func TestTokenizeMatchesComfyUI(t *testing.T) {
	v := vocab(t)
	for _, name := range []string{"short", "portrait", "weighted"} {
		raw, err := os.ReadFile(filepath.Join("..", "testdata", "krea2", "tokens", name+".json"))
		if err != nil {
			t.Skipf("no recording (%v)", err)
		}
		var want struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		}
		if err := json.Unmarshal(raw, &want); err != nil {
			t.Fatal(err)
		}
		got := Tokenize(v, want.Text)
		if len(got.IDs) != len(want.IDs) {
			t.Fatalf("%s: %d tokens, want %d\n got %v\nwant %v", name, len(got.IDs), len(want.IDs), got.IDs, want.IDs)
		}
		for i := range got.IDs {
			if got.IDs[i] != want.IDs[i] {
				t.Fatalf("%s: token %d is %d, want %d", name, i, got.IDs[i], want.IDs[i])
			}
		}
		if got.Start != 34 {
			t.Errorf("%s: conditioning starts at %d, want 34", name, got.Start)
		}
	}
}
