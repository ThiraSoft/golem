package bytebpe

import "testing"

// Every byte must survive the round trip. A byte-level tokenizer that loses
// one cannot encode the sentence that contains it.
func TestByteAlphabetRoundTrips(t *testing.T) {
	raw := make([]byte, 256)
	for i := range raw {
		raw[i] = byte(i)
	}
	encoded := encodeBytes(string(raw))
	back, err := decodeBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if string(back) != string(raw) {
		t.Fatal("a byte did not survive the round trip")
	}
}

func TestByteAlphabetKnownPoints(t *testing.T) {
	cases := []struct{ in, want string }{
		{" ", "Ġ"},
		{"\n", "Ċ"},
		{"\t", "ĉ"},
		{"a", "a"},
		{"~", "~"},
		{"hello world", "helloĠworld"},
	}
	for _, c := range cases {
		if got := encodeBytes(c.in); got != c.want {
			t.Errorf("encodeBytes(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// The alphabet is 256 distinct code points. A collision would silently merge
// two different bytes into one piece, and the sentence would come back wrong
// rather than fail.
func TestByteAlphabetIsInjective(t *testing.T) {
	seen := map[rune]byte{}
	for i := 0; i < 256; i++ {
		r := []rune(encodeBytes(string([]byte{byte(i)})))
		if len(r) != 1 {
			t.Fatalf("byte %d encoded to %d runes", i, len(r))
		}
		if prev, ok := seen[r[0]]; ok {
			t.Fatalf("bytes %d and %d both encode to %q", prev, i, r[0])
		}
		seen[r[0]] = byte(i)
	}
}

// A rune that is not in the alphabet cannot come from a piece of this
// vocabulary, and turning it into some byte anyway would be a silent wrong
// answer.
//
// Note that Latin-1 letters are *in* the alphabet: é is U+00E9, which is byte
// 233 keeping its own code point. The runes that are out are the ones above
// U+00FF, and U+00AD — the soft hyphen, the single byte the second kept range
// steps over, which is therefore given a stand-in like an unprintable one.
func TestDecodeRefusesWhatIsNotInTheAlphabet(t *testing.T) {
	for _, s := range []string{"世", "\u0500", "\u00ad"} {
		if _, err := decodeBytes(s); err == nil {
			t.Errorf("%q was decoded, but it is not in the alphabet", s)
		}
	}
	// The counterpart: a Latin-1 letter is in it, and is its own byte.
	got, err := decodeBytes("é")
	if err != nil {
		t.Fatalf("é should be in the alphabet: %v", err)
	}
	if len(got) != 1 || got[0] != 0xE9 {
		t.Errorf("é decoded to %v, want [233]", got)
	}
}

// The pieces in the file must all be spelled in this alphabet: if they are
// not, the table here is not the table the vocabulary was built with, and
// every piece would miss.
func TestEveryPieceIsSpelledInTheAlphabet(t *testing.T) {
	v := openVocab(t)
	for id := int32(0); int(id) < v.Size(); id++ {
		if v.Kind(id) != Normal {
			continue // control and user-defined pieces are literal text
		}
		if _, err := decodeBytes(v.Text(id)); err != nil {
			t.Fatalf("piece %d (%q) is not in the byte alphabet: %v", id, v.Text(id), err)
		}
	}
}
