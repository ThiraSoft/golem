package bytebpe

// The GPT-2 pre-tokenizer, which Hugging Face's ByteLevel runs when its
// use_regex is true and which ModernBERT's tokenizer declares:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// It differs from qwen2's in three places worth knowing. The contractions are
// lower case only. The optional character before a word is a space and
// nothing else, so ",b" is two words. And a number is a run of digits, not one
// digit, so "1234" is one word.
//
// The lookahead is the same as qwen2's: a run of whitespace followed by
// something gives its last character to what follows, and a lone whitespace
// character before a word that did not take it is a word of its own.

import "unicode"

// splitGPT2 cuts text into pre-merge words. The pieces concatenate back to the
// input exactly.
func splitGPT2(text string) []string {
	cpts := []rune(text)
	n := len(cpts)
	in := func(i int) bool { return i >= 0 && i < n }
	isLetter := func(i int) bool { return in(i) && unicode.IsLetter(cpts[i]) }
	isNumber := func(i int) bool { return in(i) && unicode.IsNumber(cpts[i]) }
	isSpace := func(i int) bool { return in(i) && unicode.IsSpace(cpts[i]) }
	isOther := func(i int) bool { return in(i) && !isSpace(i) && !isLetter(i) && !isNumber(i) }

	var out []string
	for pos := 0; pos < n; {
		end := pos + 1
		switch {
		case cpts[pos] == '\'' && contraction(cpts[pos+1:]) > 0:
			end = pos + 1 + contraction(cpts[pos+1:])
		case isLetter(pos) || (cpts[pos] == ' ' && isLetter(pos+1)):
			for end = pos + 1; isLetter(end); end++ {
			}
		case isNumber(pos) || (cpts[pos] == ' ' && isNumber(pos+1)):
			for end = pos + 1; isNumber(end); end++ {
			}
		case isOther(pos) || (cpts[pos] == ' ' && isOther(pos+1)):
			for end = pos + 1; isOther(end); end++ {
			}
		default:
			// Whitespace. The run is taken whole when nothing but its end
			// follows; otherwise it leaves its last character behind, unless
			// that would leave it empty.
			for end = pos + 1; isSpace(end); end++ {
			}
			if end < n && end-pos > 1 {
				end--
			}
		}
		out = append(out, string(cpts[pos:end]))
		pos = end
	}
	return out
}

// contraction is the length of the contraction's tail that follows an
// apostrophe, or zero when there is none.
func contraction(rest []rune) int {
	if len(rest) >= 2 {
		switch string(rest[:2]) {
		case "re", "ve", "ll":
			return 2
		}
	}
	if len(rest) >= 1 {
		switch rest[0] {
		case 's', 't', 'm', 'd':
			return 1
		}
	}
	return 0
}
