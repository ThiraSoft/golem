package bytebpe

// GPT-2's byte alphabet.
//
// A byte-level BPE has to see all 256 bytes, and a vocabulary is a list of
// strings — so the bytes that are not printable are given stand-ins that are.
// The 188 already safe keep their own code point; the other 68 are moved to
// U+0100 and up, in the order they come. A space becomes U+0120, which is why
// every piece in the file that starts a word starts with it.
//
// The mapping is not a choice this code makes. It is the one the vocabulary in
// the file was built with, and a different one would make every piece miss.

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

var (
	byteToRune [256]rune
	runeToByte map[rune]byte
)

func init() {
	// The three ranges that keep themselves: printable ASCII without the
	// space, then two stretches of Latin-1 that are printable too. The space
	// is deliberately not among them — it is the one byte this alphabet most
	// needs to make visible.
	kept := [256]bool{}
	for _, r := range [][2]int{{'!', '~'}, {0xA1, 0xAC}, {0xAE, 0xFF}} {
		for b := r[0]; b <= r[1]; b++ {
			kept[b] = true
		}
	}

	runeToByte = make(map[rune]byte, 256)
	next := rune(0x100)
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !kept[b] {
			r = next
			next++
		}
		byteToRune[b] = r
		runeToByte[r] = byte(b)
	}
}

// encodeBytes rewrites raw bytes as the printable stand-ins the vocabulary is
// spelled in.
func encodeBytes(s string) string {
	var b strings.Builder
	b.Grow(len(s) * 2)
	for i := 0; i < len(s); i++ {
		b.WriteRune(byteToRune[s[i]])
	}
	return b.String()
}

// decodeBytes is the inverse. A rune the alphabet does not contain is an error
// rather than a guess: it means the text did not come from a piece of this
// vocabulary, and inventing a byte for it would answer wrongly instead of
// saying so.
func decodeBytes(s string) ([]byte, error) {
	out := make([]byte, 0, utf8.RuneCountInString(s))
	for _, r := range s {
		b, ok := runeToByte[r]
		if !ok {
			return nil, fmt.Errorf("%q is not in the byte alphabet", r)
		}
		out = append(out, b)
	}
	return out, nil
}
