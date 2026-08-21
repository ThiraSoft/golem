package bytebpe

// Cutting the special tokens out of the text before anything else sees them.
//
// They have to go first, and not only for tidiness: "<|im_start|>" contains
// letters, punctuation and a vertical bar, so the pre-tokenizer would cut it
// into four words and the merges would then spell it out as ordinary text.
//
// llama.cpp scans for one special token at a time, longest first, splitting
// every raw fragment it finds it in and leaving the halves raw for the shorter
// candidates. The order matters: a short token that is a substring of a longer
// one would otherwise cut the longer one in half.

import "strings"

// fragment is either a stretch of raw text (id < 0) or a token already decided.
type fragment struct {
	text string
	id   int32
}

func (v *Vocab) partition(text string, parseSpecial bool) []fragment {
	fragments := []fragment{{text: text, id: -1}}

	for _, special := range v.specials {
		kind := v.Kind(special)
		if !parseSpecial && (kind == Control || kind == Unknown) {
			// Control and unknown tokens stay literal text. User-defined ones
			// are cut out regardless, which is what the reference does.
			continue
		}
		needle := v.Text(special)
		if needle == "" {
			continue
		}

		var next []fragment
		for _, f := range fragments {
			if f.id >= 0 {
				next = append(next, f)
				continue
			}
			rest := f.text
			for {
				at := strings.Index(rest, needle)
				if at < 0 {
					break
				}
				if at > 0 {
					next = append(next, fragment{text: rest[:at], id: -1})
				}
				next = append(next, fragment{id: special})
				rest = rest[at+len(needle):]
			}
			if rest != "" {
				next = append(next, fragment{text: rest, id: -1})
			}
		}
		fragments = next
	}
	return fragments
}
