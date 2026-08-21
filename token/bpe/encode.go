package bpe

// Encoding, as llama.cpp does it for the gemma4 pre-tokenizer.
//
// The shape is SentencePiece's outside and BPE's inside: spaces become U+2581
// so that word boundaries survive, then the merges run over raw UTF-8 — there
// is no GPT-2 byte alphabet here, and no word-level pre-splitting either. Only
// newlines cut the text, because a merge is never allowed to straddle one.

import (
	"strings"
	"unicode/utf8"

	"github.com/ThiraSoft/golem/token/merge"
	"github.com/ThiraSoft/golem/token/special"
)

// escapeSpaces replaces the ASCII space, and nothing else. The non-breaking
// space and the tab stay as they are and reach the merges as ordinary
// characters — which is what the file's own pieces expect.
func escapeSpaces(text string) string {
	return strings.ReplaceAll(text, " ", Space)
}

// splitNewlines implements the regex "[^\n]+|[\n]+": alternating runs of
// newline and of everything else, in order, nothing dropped.
func splitNewlines(text string) []string {
	var runs []string
	for i := 0; i < len(text); {
		isNewline := text[i] == '\n'
		j := i
		for j < len(text) && (text[j] == '\n') == isNewline {
			j++
		}
		runs = append(runs, text[i:j])
		i = j
	}
	return runs
}

// Encode turns text into identifiers. addBOS prepends the beginning-of-text
// piece; parseSpecial lets the control tokens in the text be recognized rather
// than spelled out.
func (v *Vocab) Encode(text string, addBOS, parseSpecial bool) []int32 {
	var out []int32
	if addBOS && v.bos >= 0 {
		out = append(out, v.bos)
	}
	for _, f := range special.Partition(text, v.specials, parseSpecial) {
		if f.ID >= 0 {
			out = append(out, f.ID)
			continue
		}
		out = v.encodeRaw(f.Text, out)
	}
	return out
}

// encodeRaw handles one stretch of ordinary text: escape, cut on newlines,
// merge each run, emit.
//
// The merger and the symbol slice are made here rather than per run: a chat
// turn is a few dozen short runs, and on those the allocation costs more than
// the merging. They belong to this call and are never shared, so Encode stays
// safe to call from several goroutines at once.
func (v *Vocab) encodeRaw(text string, out []int32) []int32 {
	var m merge.Merger
	escaped := escapeSpaces(text)
	// One symbol per rune, so the byte length is an upper bound and a generous
	// one. Sized once here, reused by every run below.
	symbols := make([]string, 0, len(escaped))
	for _, run := range splitNewlines(escaped) {
		out, symbols = v.encodeRun(&m, symbols, run, out)
	}
	return out
}

// encodeRun applies the merges to a single run and emits what survives. It
// gives the symbol slice back so the next run can reuse its capacity.
func (v *Vocab) encodeRun(m *merge.Merger, symbols []string, run string, out []int32) ([]int32, []string) {
	if run == "" {
		return out, symbols
	}
	symbols = symbols[:0]

	// A run made only of newlines that the vocabulary already knows is taken
	// whole. Splitting it would leave the merges to rebuild it, and they do
	// not always reach the same answer.
	whole := false
	if strings.Trim(run, "\n") == "" {
		if _, ok := v.ID(run); ok {
			symbols = append(symbols, run)
			whole = true
		}
	}

	// Otherwise one symbol per rune: this tokenizer merges over raw UTF-8, so
	// a character is where a symbol starts, not a byte.
	if !whole {
		for i := 0; i < len(run); {
			_, size := utf8.DecodeRuneInString(run[i:])
			symbols = append(symbols, run[i:i+size])
			i += size
		}
	}

	for _, text := range m.Apply(symbols, v.ranks) {
		out = v.emit(text, out)
	}
	return out, symbols
}

// emit writes one finished symbol. A symbol the vocabulary does not know is
// re-emitted byte by byte through the <0xNN> pieces — otherwise a single
// unexpected character would make the sentence unencodable.
func (v *Vocab) emit(text string, out []int32) []int32 {
	if id, ok := v.ID(text); ok {
		return append(out, id)
	}
	for i := 0; i < len(text); i++ {
		if id := v.byteIDs[text[i]]; id >= 0 {
			out = append(out, id)
		}
	}
	return out
}
