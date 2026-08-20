package sentencepiece

// Normalizing the text before segmentation.
//
// SentencePiece ships a precompiled character map that applies NFKC. Porting it
// would be several hundred lines for no effect on already well-formed text —
// which is everything coming out of a keyboard or a modern file. We therefore
// apply only the rules that change the segmentation in practice: whitespace
// unification, collapsing of repeats, prefix, and escaping.
//
// The limitation is real and accepted: a text containing ligatures or
// compatibility characters (ﬁ, ①, ½) would be segmented differently than under
// Python. The parity test covers a French corpus that excludes them.

import "strings"

func (t *Tokenizer) normalize(text string) string {
	var b strings.Builder
	b.Grow(len(text) + 8)

	prevSpace := false
	for _, r := range text {
		if isSpace(r) {
			if t.rules.CollapseSpaces && prevSpace {
				continue
			}
			prevSpace = true
			b.WriteByte(' ')
			continue
		}
		prevSpace = false
		b.WriteRune(r)
	}
	s := b.String()
	if t.rules.CollapseSpaces {
		s = strings.Trim(s, " ")
	}
	if s == "" {
		return ""
	}
	if t.rules.DummyPrefix {
		s = " " + s
	}
	if t.rules.EscapeSpaces {
		s = strings.ReplaceAll(s, " ", Space)
	}
	return s
}

// isSpace covers the Unicode spaces that NFKC folds to the ordinary space —
// starting with the non-breaking space, which abounds in French text before
// colons.
//
// Control characters are absent from it, and deliberately so: NFKC touches
// neither the tab nor the newline, which SentencePiece therefore encodes as
// bytes. Conflating them with the space changed the segmentation.
func isSpace(r rune) bool {
	switch r {
	case ' ', 0xA0, 0x1680,
		0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
		0x2007, 0x2008, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000:
		return true
	}
	return false
}
