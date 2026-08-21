// Package special cuts a vocabulary's special tokens out of text before the
// rest of the tokenizer sees it.
//
// It has to happen first, and not only for tidiness. "<|im_start|>" is letters,
// punctuation and vertical bars: a pre-tokenizer would cut it into four words
// and the merges would then spell it out as ordinary text, which is a
// conversation the model cannot see the shape of.
//
// The scan is llama.cpp's: one token at a time, longest first, splitting every
// raw fragment it is found in and leaving the halves raw for the shorter
// candidates. The order is what matters — a short token that is a substring of
// a longer one would otherwise cut the longer one in half.
//
// The tokens arrive as a slice rather than behind an interface. Both callers
// build one at load time, and this way there is neither a dispatch nor a
// dependency on what a Vocab happens to look like.
package special

import (
	"sort"
	"strings"
)

// Token is one special piece a text may contain.
type Token struct {
	ID   int32
	Text string
	// Hidden marks the tokens that stay literal text unless the caller asks
	// for them: the control and unknown ones. A user-defined token is cut out
	// regardless, which is what the reference does.
	Hidden bool
}

// Fragment is either a stretch of raw text (ID < 0) or a token already decided.
type Fragment struct {
	Text string
	ID   int32
}

// Sort orders tokens longest text first, which is the order Partition needs.
// Callers do this once, when the vocabulary is loaded.
func Sort(tokens []Token) {
	sort.SliceStable(tokens, func(i, j int) bool {
		return len(tokens[i].Text) > len(tokens[j].Text)
	})
}

// Partition cuts text at every special token it contains. tokens must be
// sorted longest first; parseSpecial decides whether the hidden ones are
// recognized or left as the characters they are spelled with.
func Partition(text string, tokens []Token, parseSpecial bool) []Fragment {
	fragments := []Fragment{{Text: text, ID: -1}}

	// By index and by pointer: a Token is four words, and there are a couple
	// of dozen of them per call on a path where the whole encode of a chat
	// turn is ten microseconds.
	for i := range tokens {
		token := &tokens[i]
		if token.Text == "" || (token.Hidden && !parseSpecial) {
			continue
		}

		var next []Fragment
		for _, f := range fragments {
			if f.ID >= 0 {
				next = append(next, f)
				continue
			}
			rest := f.Text
			for {
				at := strings.Index(rest, token.Text)
				if at < 0 {
					break
				}
				if at > 0 {
					next = append(next, Fragment{Text: rest[:at], ID: -1})
				}
				next = append(next, Fragment{ID: token.ID})
				rest = rest[at+len(token.Text):]
			}
			if rest != "" {
				next = append(next, Fragment{Text: rest, ID: -1})
			}
		}
		fragments = next
	}
	return fragments
}
