package bytebpe

// The qwen2 pre-tokenizer: cutting text into words before any merge runs.
//
// The file names this rule set in tokenizer.ggml.pre, and llama.cpp writes it
// down as a regex:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)
//	|[^\r\n\p{L}\p{N}]?\p{L}+
//	|\p{N}
//	| ?[^\s\p{L}\p{N}]+[\r\n]*
//	|\s*[\r\n]+
//	|\s+(?!\S)
//	|\s+
//
// but it does not run it. `\s+(?!\S)` needs a lookahead, which neither std::regex
// nor Go's RE2 has, so llama.cpp hand-writes the scan in
// unicode_regex_split_custom_qwen2 and that is what actually tokenizes every
// Qwen model. This is a transcription of that function, not of the regex above:
// where the two could differ, the function is what the vocabulary was built
// against.
//
// Three of the rules are worth reading twice, because getting them wrong costs
// a few tokens a sentence and raises nothing:
//
//   - Rule 3 is one number, with no quantifier. "1234" is four words. The
//     llama3 rules say \p{N}{1,3} and are a different tokenizer.
//   - Rule 2's optional leading character is consumed whatever it is, so a
//     space or a comma before a letter joins the word: " world" and ",b" are
//     each one word.
//   - Rule 6 is the lookahead: a run of whitespace gives its last character to
//     whatever follows, unless nothing follows. So "a   b" is "a", "  ", " b",
//     while "a   " is "a", "   ".

import "unicode"

// splitQwen2 cuts text into pre-merge words. The pieces concatenate back to
// the input exactly: nothing is dropped, nothing is added.
func splitQwen2(text string) []string {
	cpts := []rune(text)
	if len(cpts) == 0 {
		return nil
	}

	// llama.cpp reads a codepoint's category through flags that are zero when
	// the position is out of range, and several of the rules lean on that
	// rather than on a bounds check. inRange is that zero.
	inRange := func(i int) bool { return i >= 0 && i < len(cpts) }
	at := func(i int) rune {
		if !inRange(i) {
			return -1
		}
		return cpts[i]
	}
	isLetter := func(i int) bool { return inRange(i) && unicode.IsLetter(cpts[i]) }
	isNumber := func(i int) bool { return inRange(i) && unicode.IsNumber(cpts[i]) }
	isSpace := func(i int) bool { return inRange(i) && unicode.IsSpace(cpts[i]) }

	// plain is the fourth rule's test: a codepoint that is in range and is
	// none of whitespace, letter or number.
	plain := func(i int) bool {
		return inRange(i) && !isSpace(i) && !isLetter(i) && !isNumber(i)
	}

	var out []string
	prev := 0
	emit := func(end int) {
		if end > prev {
			out = append(out, string(cpts[prev:end]))
		}
		prev = end
	}

	for pos := 0; pos < len(cpts); {
		cpt := cpts[pos]

		// Rule 1: an English contraction, either case.
		if cpt == '\'' && pos+1 < len(cpts) {
			next := unicode.ToLower(at(pos + 1))
			if next == 's' || next == 't' || next == 'm' || next == 'd' {
				pos += 2
				emit(pos)
				continue
			}
			if pos+2 < len(cpts) {
				third := unicode.ToLower(at(pos + 2))
				if (next == 'r' && third == 'e') ||
					(next == 'v' && third == 'e') ||
					(next == 'l' && third == 'l') {
					pos += 3
					emit(pos)
					continue
				}
			}
		}

		// Rule 2: one optional non-letter, non-digit, then letters. The
		// leading character is taken whatever it is — including a letter, in
		// which case it is simply the first of them.
		if cpt != '\r' && cpt != '\n' && !isNumber(pos) {
			if isLetter(pos) || isLetter(pos+1) {
				pos++
				for isLetter(pos) {
					pos++
				}
				emit(pos)
				continue
			}
		}

		// Rule 3: exactly one number.
		if isNumber(pos) {
			pos++
			emit(pos)
			continue
		}

		// Rule 4: an optional space, then a run of characters that are neither
		// whitespace nor letters nor digits, then any line endings that follow.
		probe := pos
		if cpt == ' ' {
			probe = pos + 1
		}
		if plain(probe) {
			if cpt == ' ' {
				pos++
			}
			for plain(pos) {
				pos++
			}
			for at(pos) == '\r' || at(pos) == '\n' {
				pos++
			}
			emit(pos)
			continue
		}

		// The three whitespace rules share one measurement: how far the run
		// goes, and where the last line ending inside it ended.
		spaces := 0
		afterLineEnd := 0
		for isSpace(pos + spaces) {
			if c := cpts[pos+spaces]; c == '\r' || c == '\n' {
				afterLineEnd = pos + spaces + 1
			}
			spaces++
		}

		// Rule 5: whitespace ending in a line ending — the run stops after the
		// last one, so trailing spaces on the next line start a new word.
		if afterLineEnd > 0 {
			pos = afterLineEnd
			emit(pos)
			continue
		}

		// Rule 6: whitespace with something after it gives its last character
		// away, so that the following word keeps its leading space.
		if spaces > 1 && inRange(pos+spaces) {
			pos += spaces - 1
			emit(pos)
			continue
		}

		// Rule 7: whitespace with nothing after it, taken whole.
		if spaces > 0 {
			pos += spaces
			emit(pos)
			continue
		}

		// Nothing matched, which the rules above make almost unreachable: one
		// codepoint, so that the scan always moves.
		pos++
		emit(pos)
	}
	return out
}
