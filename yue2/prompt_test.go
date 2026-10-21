package yue2

import (
	"reflect"
	"testing"
)

func TestPrompt(t *testing.T) {
	f := loadFixtures(t)
	if !f.has("prompt/guerreros_full_with_abc") || !f.has("prompt/guerreros_melody_no_abc") || !f.has("prompt/guerreros_off") {
		t.Skip("prompt fixtures not found: see ref/yue2/README.md")
	}

	tok := loadTestTokenizer(t)
	req := loadTestReplay(t)
	abcTokens := tok.Encode(req.ABC)

	// 1. Cot parsing and defaults
	cot, err := ParseCot("full")
	if err != nil || cot != CotFull {
		t.Fatalf("ParseCot(full): got %v, err %v", cot, err)
	}
	cot, err = ParseCot("melody")
	if err != nil || cot != CotMelody {
		t.Fatalf("ParseCot(melody): got %v, err %v", cot, err)
	}
	cot, err = ParseCot("off")
	if err != nil || cot != CotOff {
		t.Fatalf("ParseCot(off): got %v, err %v", cot, err)
	}
	if _, err := ParseCot("invalid"); err == nil {
		t.Errorf("ParseCot(invalid) expected error, got nil")
	}

	if g := DefaultGuidance(CotOff); g != 1.01 {
		t.Errorf("DefaultGuidance(CotOff) = %g, want 1.01", g)
	}
	if g := DefaultGuidance(CotFull); g != 1.0 {
		t.Errorf("DefaultGuidance(CotFull) = %g, want 1.0", g)
	}
	if g := DefaultGuidance(CotMelody); g != 1.0 {
		t.Errorf("DefaultGuidance(CotMelody) = %g, want 1.0", g)
	}

	// 2. Full conditional prompt with ABC
	gotFull := BuildPrompt(tok, CotFull, req.Style, req.Lyrics, abcTokens, true)
	wantFull := f.readInt32(t, "prompt/guerreros_full_with_abc")
	if !reflect.DeepEqual(gotFull, wantFull) {
		t.Errorf("prompt/guerreros_full_with_abc mismatch: got %d tokens, want %d tokens", len(gotFull), len(wantFull))
	}

	// 3. Melody prompt without ABC (stops at ABCStart)
	gotMelody := BuildPrompt(tok, CotMelody, req.Style, req.Lyrics, nil, false)
	wantMelody := f.readInt32(t, "prompt/guerreros_melody_no_abc")
	if !reflect.DeepEqual(gotMelody, wantMelody) {
		t.Errorf("prompt/guerreros_melody_no_abc mismatch: got %d tokens, want %d tokens", len(gotMelody), len(wantMelody))
	}

	// 4. Cot off prompt
	gotOff := BuildPrompt(tok, CotOff, req.Style, req.Lyrics, nil, false)
	wantOff := f.readInt32(t, "prompt/guerreros_off")
	if !reflect.DeepEqual(gotOff, wantOff) {
		t.Errorf("prompt/guerreros_off mismatch: got %d tokens, want %d tokens", len(gotOff), len(wantOff))
	}

	// 5. Unconditional prompts
	if f.has("prompt/negative_off") {
		gotNegOff := BuildNegative(tok, CotOff, nil, false)
		wantNegOff := f.readInt32(t, "prompt/negative_off")
		if !reflect.DeepEqual(gotNegOff, wantNegOff) {
			t.Errorf("prompt/negative_off mismatch: got %d tokens, want %d tokens", len(gotNegOff), len(wantNegOff))
		}
	}

	if f.has("prompt/negative_full_with_abc") {
		gotNegFull := BuildNegative(tok, CotFull, abcTokens, true)
		wantNegFull := f.readInt32(t, "prompt/negative_full_with_abc")
		if !reflect.DeepEqual(gotNegFull, wantNegFull) {
			t.Errorf("prompt/negative_full_with_abc mismatch: got %d tokens, want %d tokens", len(gotNegFull), len(wantNegFull))
		}
	}
}
