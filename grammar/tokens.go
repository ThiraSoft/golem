package grammar

// Tokens is what a grammar needs to know about a vocabulary: the bytes of every
// token, the first of those bytes, and which identifiers end a turn.
//
// It is built once per model and shared by every grammar over it. Decoding a
// piece per token per step is the cost that sinks a naive port — a quarter of a
// million string conversions for one drawn token — and this is where it is paid
// instead, once.
//
// The bytes are kept raw rather than as text: a byte-level BPE splits a
// character across two tokens whenever it pleases, and a piece is not always
// valid UTF-8 on its own. What makes sense of it is the partial state a
// Grammar carries between tokens.
type Tokens struct {
	bytes [][]byte
	first []byte // the first byte of each piece, 0 when there is none
	eog   []bool
}

// NewTokens reads the vocabulary once. size is the width of a logits row, which
// is the number of identifiers a model can name.
func NewTokens(size int, piece func(int32) string, isEOG func(int32) bool) *Tokens {
	t := &Tokens{
		bytes: make([][]byte, size),
		first: make([]byte, size),
		eog:   make([]bool, size),
	}
	for id := 0; id < size; id++ {
		b := []byte(piece(int32(id)))
		t.bytes[id] = b
		if len(b) > 0 {
			t.first[id] = b[0]
		}
		t.eog[id] = isEOG(int32(id))
	}
	return t
}

// Size is how many identifiers the table holds.
func (t *Tokens) Size() int { return len(t.bytes) }

// partialUTF8 is a character cut in two by the tokenizer: the bits read so far
// and how many bytes are still missing. A remain of -1 is a sequence that
// cannot be completed at all.
type partialUTF8 struct {
	value  uint32
	remain int
}

// utf8Remaining says how many bytes follow a lead byte, by its high nibble.
// llama-grammar.cpp:36 writes the same table; a zero is a continuation byte
// where a lead byte was expected, which makes remain negative and poisons the
// sequence.
var utf8Remaining = [16]int{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4}

// decodeUTF8 reads a piece into code points, continuing whatever character the
// piece before it left open, and says what this one leaves open in turn.
func decodeUTF8(src []byte, partial partialUTF8) ([]rune, partialUTF8) {
	out := make([]rune, 0, len(src))
	value, remain := partial.value, partial.remain
	pos := 0

	for pos < len(src) && remain > 0 {
		b := src[pos]
		if b>>6 != 2 {
			// A byte that is not a continuation, where one was owed.
			return out, partialUTF8{0, -1}
		}
		value = value<<6 + uint32(b&0x3F)
		pos++
		remain--
	}
	if partial.remain > 0 && remain == 0 {
		out = append(out, rune(value))
	}

	for pos < len(src) {
		lead := src[pos]
		remain = utf8Remaining[lead>>4] - 1
		if remain < 0 {
			return out[:0], partialUTF8{0, remain}
		}
		mask := byte(1<<(7-remain)) - 1
		value = uint32(lead & mask)
		pos++
		for pos < len(src) && remain > 0 {
			value = value<<6 + uint32(src[pos]&0x3F)
			pos++
			remain--
		}
		if remain == 0 {
			out = append(out, rune(value))
		}
	}
	return out, partialUTF8{value, remain}
}
