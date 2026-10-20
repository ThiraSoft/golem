package yue2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
	"golang.org/x/text/unicode/norm"
)

func loadTestTokenizer(t testing.TB) *Tokenizer {
	t.Helper()
	modelPath := filepath.Join(weightsDir(), "YuE2-3B-Q8_0.gguf")
	g, err := tensors.OpenGGUF(modelPath)
	if err != nil {
		t.Skipf("cannot open YuE2 GGUF at %s: %v", modelPath, err)
	}
	t.Cleanup(func() { g.Close() })

	tok, err := NewTokenizer(g)
	if err != nil {
		t.Fatalf("NewTokenizer: %v", err)
	}
	return tok
}

type testReplayRequest struct {
	Style  string `json:"style"`
	Lyrics string `json:"lyrics"`
	ABC    string `json:"abc"`
	LMSeed int64  `json:"lm_seed"`
	Seed   uint64 `json:"seed"`
	Cot    string `json:"cot"`
}

func loadTestReplay(t testing.TB) testReplayRequest {
	t.Helper()
	replayPath := filepath.Join("..", "testdata", "yue2", "bench", "guerreros-replay.json")
	data, err := os.ReadFile(replayPath)
	if err != nil {
		replayPath = "/mnt/data/golem-testdata/yue2/bench/guerreros-replay.json"
		data, err = os.ReadFile(replayPath)
		if err != nil {
			t.Fatalf("cannot read replay json at %s: %v", replayPath, err)
		}
	}
	var req testReplayRequest
	if err := json.Unmarshal(data, &req); err != nil {
		t.Fatalf("parse replay json: %v", err)
	}
	return req
}

func TestTokenizerFixtures(t *testing.T) {
	f := loadFixtures(t)
	if !f.has("tokens/guerreros_request") || !f.has("tokens/bpe_text") || !f.has("tokens/guerreros_lyrics_nfd") {
		t.Skip("token fixtures not found: see ref/yue2/README.md")
	}

	tok := loadTestTokenizer(t)
	req := loadTestReplay(t)

	// 1. Guerreros request text
	reqText := instruction(CotFull) + "\n[Tags]\n" + req.Style + "\n[Lyrics]\n" + req.Lyrics + "\n"
	gotReq := tok.Encode(reqText)
	wantReq := f.readInt32(t, "tokens/guerreros_request")
	if !reflect.DeepEqual(gotReq, wantReq) {
		t.Errorf("guerreros_request mismatch: got %d tokens, want %d tokens", len(gotReq), len(wantReq))
	}

	// 2. bpe-text.txt
	bpeTextPath := "/mnt/data/dev/musicbench/yue2/tests/bpe-text.txt"
	bpeTextRaw, err := os.ReadFile(bpeTextPath)
	if err != nil {
		t.Skipf("cannot read %s: %v", bpeTextPath, err)
	}
	gotBPE := tok.Encode(string(bpeTextRaw))
	wantBPE := f.readInt32(t, "tokens/bpe_text")
	if !reflect.DeepEqual(gotBPE, wantBPE) {
		t.Errorf("bpe_text mismatch: got %d tokens, want %d tokens", len(gotBPE), len(wantBPE))
	}

	// 3. Guerreros lyrics in NFD
	nfdLyrics := norm.NFD.String(req.Lyrics)
	gotNFD := tok.Encode(nfdLyrics)
	wantNFD := f.readInt32(t, "tokens/guerreros_lyrics_nfd")
	if !reflect.DeepEqual(gotNFD, wantNFD) {
		t.Errorf("guerreros_lyrics_nfd mismatch: got %d tokens, want %d tokens", len(gotNFD), len(wantNFD))
	}

	// 4. Verify NFD case equals NFC case
	gotNFC := tok.Encode(req.Lyrics)
	if !reflect.DeepEqual(gotNFD, gotNFC) {
		t.Errorf("NFD case does not equal NFC case: NFD len %d != NFC len %d", len(gotNFD), len(gotNFC))
	}

	// 5. Guerreros ABC tokens
	if f.has("tokens/guerreros_abc") {
		gotABC := tok.Encode(req.ABC)
		wantABC := f.readInt32(t, "tokens/guerreros_abc")
		if !reflect.DeepEqual(gotABC, wantABC) {
			t.Errorf("guerreros_abc mismatch: got %d tokens, want %d tokens", len(gotABC), len(wantABC))
		}

		decodedABC := tok.Decode(gotABC)
		if decodedABC != req.ABC {
			t.Errorf("ABC decode roundtrip mismatch: got len %d, want len %d", len(decodedABC), len(req.ABC))
		}
	}
}
