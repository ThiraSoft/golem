// Package merge applies a ranked byte-pair merge table to a list of symbols.
//
// This is the middle of every BPE tokenizer and the whole of none of them. What
// differs between one and the next is at the two ends: how the text becomes the
// first list of symbols — SentencePiece escapes its spaces, GPT-2 maps its
// bytes onto printable stand-ins and cuts words apart first — and what happens
// to a symbol the vocabulary does not know once the merging stops. Between
// those two, the loop is the same loop, and it is llama.cpp's.
//
// The table is passed as a map rather than behind an interface. The rank lookup
// is the hot operation here — it runs once per adjacent pair and twice more for
// every merge applied — and an interface would put an indirect call inside it
// for no gain in expressiveness: every caller already holds exactly this map.
//
// A Merger carries the three buffers the loop needs. Text arrives as many short
// runs rather than one long one, and allocating per run costs more than the
// merging does on the short ones: keeping the buffers is worth the three lines
// it takes. A Merger belongs to one call and is not safe to share between
// goroutines, which is why it is the caller's and not the vocabulary's.
package merge

import "container/heap"

// Pair is two symbol texts, as they appear on one line of the merge table.
type Pair struct{ Left, Right string }

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

// Merger holds the buffers one merge loop needs, so that a caller working
// through many short runs pays for them once.
//
// The zero value is ready. It is not safe for concurrent use.
type Merger struct {
	list []symbol
	q    queue
	out  []string
}

// Apply merges the symbols as far as the table allows and returns what
// survives, in order.
//
// The result aliases the Merger's buffer and stays valid only until the next
// call on the same Merger. A caller that needs to keep it must copy it.
func (m *Merger) Apply(symbols []string, ranks map[Pair]int) []string {
	m.out = m.out[:0]
	if len(symbols) == 0 {
		return m.out
	}
	if len(symbols) == 1 {
		return append(m.out, symbols[0])
	}

	// Sized once from the input rather than grown by appending: the buffers
	// are reused across runs, so the first run pays and the rest do not.
	if cap(m.list) < len(symbols) {
		m.list = make([]symbol, 0, len(symbols))
	}
	if cap(m.out) < len(symbols) {
		m.out = make([]string, 0, len(symbols))
	}
	m.list = m.list[:0]
	for i, text := range symbols {
		next := i + 1
		if next == len(symbols) {
			next = -1
		}
		m.list = append(m.list, symbol{text: text, prev: i - 1, next: next})
	}
	list := m.list

	m.q = m.q[:0]
	q := &m.q
	push := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		rank, ok := ranks[Pair{list[left].text, list[right].text}]
		if !ok {
			return
		}
		heap.Push(q, bigram{
			left:  left,
			right: right,
			text:  list[left].text + list[right].text,
			rank:  rank,
		})
	}
	for i := 1; i < len(list); i++ {
		push(i-1, i)
	}

	for q.Len() > 0 {
		b := heap.Pop(q).(bigram)
		left, right := &list[b.left], &list[b.right]
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
			list[right.next].prev = b.left
		}

		push(left.prev, b.left)
		push(b.left, left.next)
	}

	for i := range list {
		if list[i].text == "" {
			continue
		}
		m.out = append(m.out, list[i].text)
	}
	return m.out
}
