package text

// Shaping the text before synthesis.
//
// The model was trained on properly written sentences: a capital at the start,
// punctuation at the end. Giving it anything else does not raise an error, but
// it produces hesitant prosody and sometimes swallowed words. These few rules
// are therefore exactly those of the Python daemon.

import (
	"strings"
	"unicode"
)

// Rules are the per-language text preparation flags. Both come straight from
// the language's config: the models were trained under them, and applying the
// wrong set costs prosody rather than correctness.
type Rules struct {
	// DropSemicolons folds ";" into ",".
	DropSemicolons bool
	// PadShortInputs prefixes eight spaces to inputs of fewer than five words,
	// which the model needs in order not to rush them.
	PadShortInputs bool
}

// Prepare normalizes a sentence for synthesis.
func Prepare(t string, r Rules) string {
	t = strings.TrimSpace(t)
	t = strings.NewReplacer("\n", " ", "\r", " ", "  ", " ").Replace(t)
	if r.DropSemicolons {
		t = strings.ReplaceAll(t, ";", ",")
	}
	if t == "" {
		return ""
	}

	runes := []rune(t)
	if !unicode.IsUpper(runes[0]) {
		runes[0] = unicode.ToUpper(runes[0])
	}
	// A sentence ending on a letter leaves the model without an end-of-sentence
	// signal; it then keeps generating into the void.
	if last := runes[len(runes)-1]; unicode.IsLetter(last) || unicode.IsDigit(last) {
		runes = append(runes, '.')
	}
	out := string(runes)

	// A handful of tokens is not enough for the model to settle into a rhythm;
	// the padding buys it some. Only english_2026-01 asks for this.
	if r.PadShortInputs && len(strings.Fields(out)) < 5 {
		out = strings.Repeat(" ", 8) + out
	}
	return out
}

// Segment splits a text into chunks that fit under `maxTokens`.
//
// Past fifty tokens or so, the model starts skipping words: it has to be split,
// and split at the boundaries the voice would respect anyway — strong
// punctuation first, commas next. Each chunk is then synthesized from the same
// voice state.
func Segment(t string, maxTokens int, count func(string) int, r Rules) []string {
	t = Prepare(t, r)
	if t == "" {
		return nil
	}

	sentences := splitAfter(t, ".!?…")
	var refined []string
	for _, s := range sentences {
		if count(s) <= maxTokens {
			refined = append(refined, s)
			continue
		}
		sub := splitAfter(s, ",;:")
		if len(sub) > 1 {
			refined = append(refined, sub...)
		} else {
			refined = append(refined, s)
		}
	}

	var chunks []string
	current, currentTokens := "", 0
	for _, s := range refined {
		n := count(s)
		switch {
		case current == "":
			current, currentTokens = s, n
		case currentTokens+n > maxTokens:
			chunks = append(chunks, strings.TrimSpace(current))
			current, currentTokens = s, n
		default:
			current += " " + s
			currentTokens += n
		}
	}
	if current != "" {
		chunks = append(chunks, strings.TrimSpace(current))
	}
	return chunks
}

// splitAfter splits after each run of terminal marks, keeping them attached to
// what precedes them.
func splitAfter(t string, marks string) []string {
	var segments []string
	start := 0
	prevIsMark := false
	for i, r := range t {
		if strings.ContainsRune(marks, r) {
			prevIsMark = true
			continue
		}
		if prevIsMark && r != ' ' {
			segments = append(segments, strings.TrimSpace(t[start:i]))
			start = i
		}
		prevIsMark = false
	}
	if rest := strings.TrimSpace(t[start:]); rest != "" {
		segments = append(segments, rest)
	}
	return segments
}
