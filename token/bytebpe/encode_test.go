package bytebpe

import "testing"

// Round-tripping is the assertion that needs no fixture: whatever the
// identifiers turn out to be, decoding them must give the text back exactly.
func TestEncodeRoundTrips(t *testing.T) {
	v := openVocab(t)

	for _, text := range []string{
		"",
		"The capital of France is",
		"hello world",
		"L'élève a mangé une crêpe à Nîmes.",
		"日本語のテキストです。",
		"1234567890 and 42 and 3.14159",
		"bravo 🎉🇫🇷 et voilà",
		"func main() {\n\tfmt.Println(\"hi\")\n}",
		"  leading and trailing  ",
		"a\n\n\nb\r\nc\t\td",
		"Привет, мир — hello 世界",
		"\xef\xbf\xbd unlikely \xe2\x80\x8b zero width",
	} {
		ids := v.Encode(text, false, false)
		if text != "" && len(ids) == 0 {
			t.Errorf("Encode(%q) produced nothing", text)
			continue
		}
		if back := v.Decode(ids, false); back != text {
			t.Errorf("Encode/Decode of %q gave %q", text, back)
		}
	}
}

// Nothing is prepended unless asked, and this file asks for nothing.
func TestEncodeAddsNoBOS(t *testing.T) {
	v := openVocab(t)

	plain := v.Encode("hello", false, false)
	asked := v.Encode("hello", true, false)
	if v.BOS() < 0 {
		if len(asked) != len(plain) {
			t.Error("a file with no BOS token prepended something anyway")
		}
		return
	}
	if len(asked) != len(plain)+1 || asked[0] != v.BOS() {
		t.Errorf("addBOS produced %v against %v", asked, plain)
	}
}

func TestSpecialTokensAreParsedOnlyWhenAsked(t *testing.T) {
	v := openVocab(t)
	const text = "before <|im_end|> after"

	parsed := v.Encode(text, false, true)
	literal := v.Encode(text, false, false)

	if len(parsed) >= len(literal) {
		t.Fatalf("parse_special gave %d identifiers against %d without it; recognising a marker should produce fewer",
			len(parsed), len(literal))
	}

	end, ok := v.ID("<|im_end|>")
	if !ok {
		t.Fatal("this vocabulary has no <|im_end|>")
	}
	var found bool
	for _, id := range parsed {
		if id == end {
			found = true
		}
	}
	if !found {
		t.Error("parse_special did not recognise <|im_end|>")
	}
	for _, id := range literal {
		if id == end {
			t.Error("<|im_end|> was recognised without parse_special")
		}
	}

	// Either way the text comes back whole, which is what says the marker was
	// spelled out rather than dropped.
	if back := v.Decode(literal, true); back != text {
		t.Errorf("literal decode gave %q", back)
	}
	if back := v.Decode(parsed, true); back != text {
		t.Errorf("parsed decode gave %q", back)
	}
}

// A control piece is invisible in a transcript and present in a dump.
func TestPieceHidesControlTokensUnlessAsked(t *testing.T) {
	v := openVocab(t)

	id, ok := v.ID("<|im_end|>")
	if !ok {
		t.Fatal("this vocabulary has no <|im_end|>")
	}
	if got := v.Piece(id, false); got != "" {
		t.Errorf("Piece(%d, false) = %q, want nothing", id, got)
	}
	if got := v.Piece(id, true); got != "<|im_end|>" {
		t.Errorf("Piece(%d, true) = %q", id, got)
	}
}

// The stand-ins come back as the bytes they stand for.
func TestPieceDecodesTheAlphabet(t *testing.T) {
	v := openVocab(t)

	id, ok := v.ID("Ġthe")
	if !ok {
		t.Skip("this vocabulary has no Ġthe")
	}
	if got := v.Piece(id, false); got != " the" {
		t.Errorf("Piece = %q, want %q", got, " the")
	}
}
