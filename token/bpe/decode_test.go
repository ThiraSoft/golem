package bpe

import "testing"

func TestPiece(t *testing.T) {
	v := openVocab(t)

	cases := []struct {
		id      int32
		special bool
		want    string
	}{
		{506, false, " the"}, // ▁the, unescaped
		{107, false, "\n"},   // a normal token that is a newline
		{248, false, "\n"},   // <0x0A>, the byte
		{1, false, ""},       // <eos> is a control token: nothing, by default
		{1, true, "<eos>"},   // unless specials are asked for
		{106, true, "<turn|>"},
	}
	for _, c := range cases {
		if got := v.Piece(c.id, c.special); got != c.want {
			t.Errorf("Piece(%d, %v) = %q, want %q", c.id, c.special, got, c.want)
		}
	}
}

// Decoding what llama.cpp decoded from the same identifiers.
func TestDecodeMatchesReference(t *testing.T) {
	v := openVocab(t)

	for _, c := range loadCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			if got := v.Decode(c.IDs, true); got != c.Detokenized {
				t.Fatalf("ids %v\n got %q\nwant %q", c.IDs, got, c.Detokenized)
			}
		})
	}
}

// The end of a turn is a property of the vocabulary, and the generation loop
// will have nowhere else to ask.
func TestIsEOG(t *testing.T) {
	v := openVocab(t)

	for _, text := range []string{"<eos>", "<turn|>"} {
		id, ok := v.ID(text)
		if !ok {
			t.Fatalf("%q is missing from the vocabulary", text)
		}
		if !v.IsEOG(id) {
			t.Errorf("%q should end a turn", text)
		}
	}
	if id, _ := v.ID("▁the"); v.IsEOG(id) {
		t.Error("an ordinary piece should not end a turn")
	}
}
