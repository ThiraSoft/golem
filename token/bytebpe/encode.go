package bytebpe

// Encoding, as llama.cpp does it for the qwen2 pre-tokenizer.
//
// Three stages, and each one is somewhere else: the special tokens are cut out
// in special.go, the rest is cut into words in split.go, and each word is
// merged by token/merge. What is left here is the order they run in and the
// one thing between them — every word goes through the byte alphabet before a
// merge sees it, because that is the alphabet the vocabulary is spelled in.
//
// There is no byte fallback at the end and no need for one: every single byte
// has a piece, so a word always resolves.

import (
	"github.com/ThiraSoft/golem/token/merge"
	"github.com/ThiraSoft/golem/token/special"
)

// Encode turns text into identifiers. addBOS prepends the beginning-of-text
// piece, which this family of models declares it does not want; parseSpecial
// lets the control tokens in the text be recognized rather than spelled out.
func (v *Vocab) Encode(text string, addBOS, parseSpecial bool) []int32 {
	var out []int32
	if addBOS && v.bos >= 0 {
		out = append(out, v.bos)
	}

	// The merger and the symbol slice are made once per call rather than per
	// word: a sentence is dozens of short words, and on those the allocation
	// costs more than the merging. They belong to this call and are never
	// shared, so Encode stays safe to call from several goroutines at once.
	var m merge.Merger
	symbols := make([]string, 0, len(text))

	for _, f := range special.Partition(text, v.specials, parseSpecial) {
		if f.ID >= 0 {
			out = append(out, f.ID)
			continue
		}
		for _, word := range splitQwen2(f.Text) {
			out, symbols = v.encodeWord(&m, symbols, word, out)
		}
	}
	return out
}

// encodeWord merges one pre-split word and emits what survives. It gives the
// symbol slice back so the next word can reuse its capacity.
func (v *Vocab) encodeWord(m *merge.Merger, symbols []string, word string, out []int32) ([]int32, []string) {
	if word == "" {
		return out, symbols
	}
	symbols = symbols[:0]

	// One symbol per stand-in rune, which is one per byte of the original: the
	// merges are over the alphabet, not over UTF-8.
	for _, r := range encodeBytes(word) {
		symbols = append(symbols, string(r))
	}

	for _, text := range m.Apply(symbols, v.ranks) {
		id, ok := v.ID(text)
		if !ok {
			// Unreachable: the merges only ever join pieces the table names,
			// and every single stand-in is a piece. Saying so is better than
			// dropping a character quietly.
			panic("bytebpe: merged symbol " + text + " is not in the vocabulary")
		}
		out = append(out, id)
	}
	return out, symbols
}
