package bpe

// Encoding, as llama.cpp does it for the gemma4 pre-tokenizer.
//
// The shape is SentencePiece's outside and BPE's inside: spaces become U+2581
// so that word boundaries survive, then the merges run over raw UTF-8 — there
// is no GPT-2 byte alphabet here, and no word-level pre-splitting either. Only
// newlines cut the text, because a merge is never allowed to straddle one.

import (
	"container/heap"
	"strings"
	"unicode/utf8"
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

// symbol is one span of the run being merged, in a doubly-linked list held in
// a slice. A merged-away symbol keeps its place with an empty text rather than
// being removed, so that the indices already queued stay valid.
type symbol struct {
	text       string
	prev, next int
}

// bigram is a candidate merge waiting in the queue. text is what the pair
// spelled when it was queued: a merge on either side may have changed that,
// and the pop checks it before applying.
type bigram struct {
	left, right int
	text        string
	rank        int
}

// queue orders by ascending rank, ties by ascending left index — llama.cpp's
// comparator, which decides the outcome whenever two merges are equally good.
type queue []bigram

func (q queue) Len() int { return len(q) }
func (q queue) Less(i, j int) bool {
	if q[i].rank != q[j].rank {
		return q[i].rank < q[j].rank
	}
	return q[i].left < q[j].left
}
func (q queue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *queue) Push(x any)   { *q = append(*q, x.(bigram)) }
func (q *queue) Pop() any {
	old := *q
	n := len(old)
	item := old[n-1]
	*q = old[:n-1]
	return item
}

// Encode turns text into identifiers. addBOS prepends the beginning-of-text
// piece; parseSpecial lets the control tokens in the text be recognized rather
// than spelled out.
func (v *Vocab) Encode(text string, addBOS, parseSpecial bool) []int32 {
	var out []int32
	if addBOS && v.bos >= 0 {
		out = append(out, v.bos)
	}
	for _, f := range v.partition(text, parseSpecial) {
		if f.id >= 0 {
			out = append(out, f.id)
			continue
		}
		out = v.encodeRaw(f.text, out)
	}
	return out
}

// encodeRaw handles one stretch of ordinary text: escape, cut on newlines,
// merge each run, emit.
func (v *Vocab) encodeRaw(text string, out []int32) []int32 {
	for _, run := range splitNewlines(escapeSpaces(text)) {
		out = v.encodeRun(run, out)
	}
	return out
}

// encodeRun applies the merges to a single run and emits what survives.
func (v *Vocab) encodeRun(run string, out []int32) []int32 {
	if run == "" {
		return out
	}

	var symbols []symbol

	// A run made only of newlines that the vocabulary already knows is taken
	// whole. Splitting it would leave the merges to rebuild it, and they do
	// not always reach the same answer.
	if strings.Trim(run, "\n") == "" {
		if _, ok := v.ID(run); ok {
			symbols = []symbol{{text: run, prev: -1, next: -1}}
		}
	}

	if symbols == nil {
		for i := 0; i < len(run); {
			_, size := utf8.DecodeRuneInString(run[i:])
			index := len(symbols)
			next := index + 1
			if i+size == len(run) {
				next = -1
			}
			symbols = append(symbols, symbol{text: run[i : i+size], prev: index - 1, next: next})
			i += size
		}
	}

	q := &queue{}
	push := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		rank, ok := v.rank(symbols[left].text, symbols[right].text)
		if !ok {
			return
		}
		heap.Push(q, bigram{
			left:  left,
			right: right,
			text:  symbols[left].text + symbols[right].text,
			rank:  rank,
		})
	}
	for i := 1; i < len(symbols); i++ {
		push(i-1, i)
	}

	for q.Len() > 0 {
		b := heap.Pop(q).(bigram)
		left, right := &symbols[b.left], &symbols[b.right]
		if left.text == "" || right.text == "" {
			continue
		}
		if left.text+right.text != b.text {
			continue // the pair has moved on since it was queued
		}

		left.text += right.text
		right.text = ""
		left.next = right.next
		if right.next >= 0 {
			symbols[right.next].prev = b.left
		}

		push(left.prev, b.left)
		push(b.left, left.next)
	}

	for i := range symbols {
		if symbols[i].text == "" {
			continue
		}
		out = v.emit(symbols[i].text, out)
	}
	return out
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
