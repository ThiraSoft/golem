// Package ugm is SentencePiece's unigram tokenizer as llama.cpp runs it for a
// vocabulary it calls "t5": XLM-RoBERTa's, and therefore nomic-embed-text-v2's.
//
// token/sentencepiece is the same algorithm read from a .model file, and it
// skips the precompiled character map on the grounds that keyboard text does
// not need it. That is not true of this vocabulary. The map is what turns a
// newline and a tab into a space, folds ﬁ and ① and full-width letters under
// NFKC, and composes an e and a combining accent into é, and a multilingual
// embedder is handed all of those. So this package ports llama.cpp's
// llm_tokenizer_ugm as it is: the map walked as the XOR-compressed double
// array the file stores, user-defined pieces matched before it, a Viterbi pass
// over code points, and a run of unknown characters answered by one <unk>.
package ugm

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/ThiraSoft/golem/tensors"
)

// The token types a GGUF writes, which are SentencePiece's.
const (
	typeNormal      = 1
	typeUnknown     = 2
	typeControl     = 3
	typeUserDefined = 4
	typeUnused      = 5
	typeByte        = 6
)

// escapedSpace is what the vocabulary writes a space as.
const escapedSpace = "▁"

// Tokenizer turns text into identifiers.
type Tokenizer struct {
	pieces []string
	scores []float32
	types  []int32

	trie        trie // normal, user-defined and unused pieces
	userDefined trie // user-defined pieces alone, matched before the map

	// special is every control, unknown and user-defined piece, longest
	// first, which is the order llama.cpp splits the text on them in.
	special []int

	charsmap charsmap

	unknownScore float64

	BOS, EOS, Unknown int
	addBOS, addEOS    bool

	addSpacePrefix, removeExtraSpaces, escapeSpaces bool
}

// Load reads the vocabulary out of a GGUF.
func Load(g *tensors.GGUF) (*Tokenizer, error) {
	model, err := g.String("tokenizer.ggml.model")
	if err != nil {
		return nil, err
	}
	if model != "t5" {
		return nil, fmt.Errorf("ugm: the vocabulary is %q, not a unigram (t5) one", model)
	}
	pieces, err := g.Strings("tokenizer.ggml.tokens")
	if err != nil {
		return nil, err
	}
	scores, err := floatArray(g, "tokenizer.ggml.scores")
	if err != nil {
		return nil, err
	}
	types, err := g.Uint32Slice("tokenizer.ggml.token_type")
	if err != nil {
		return nil, err
	}
	if len(scores) != len(pieces) || len(types) != len(pieces) {
		return nil, fmt.Errorf("ugm: %d pieces, %d scores and %d types", len(pieces), len(scores), len(types))
	}

	t := &Tokenizer{
		pieces: pieces,
		scores: scores,
		types:  make([]int32, len(types)),
		// llama.cpp's defaults for a t5 vocabulary, which the file overrides
		// key by key below.
		BOS: -1, EOS: 1, Unknown: 2,
		addBOS: false, addEOS: true,
		addSpacePrefix: true, escapeSpaces: true,
	}
	for i, v := range types {
		t.types[i] = int32(v)
	}
	id := func(key string, dst *int) {
		if v, err := g.Uint32(key); err == nil {
			*dst = int(v)
		}
	}
	id("tokenizer.ggml.bos_token_id", &t.BOS)
	id("tokenizer.ggml.eos_token_id", &t.EOS)
	id("tokenizer.ggml.unknown_token_id", &t.Unknown)
	flag := func(key string, dst *bool) {
		if v, err := g.Bool(key); err == nil {
			*dst = v
		}
	}
	flag("tokenizer.ggml.add_bos_token", &t.addBOS)
	flag("tokenizer.ggml.add_eos_token", &t.addEOS)
	flag("tokenizer.ggml.add_space_prefix", &t.addSpacePrefix)
	flag("tokenizer.ggml.remove_extra_whitespaces", &t.removeExtraSpaces)

	if raw, ok := g.Meta["tokenizer.ggml.precompiled_charsmap"]; ok {
		blob, err := byteArray(raw)
		if err != nil {
			return nil, err
		}
		if t.charsmap, err = parseCharsmap(blob); err != nil {
			return nil, err
		}
	}

	minScore := math.Inf(1)
	for i, p := range pieces {
		switch t.types[i] {
		case typeNormal:
			minScore = math.Min(minScore, float64(scores[i]))
			t.trie.insert(p, i)
		case typeUserDefined:
			t.trie.insert(p, i)
			t.userDefined.insert(p, i)
		case typeUnused:
			t.trie.insert(p, i)
		}
		switch t.types[i] {
		case typeControl, typeUnknown, typeUserDefined:
			if p != "" {
				t.special = append(t.special, i)
			}
		}
	}
	// llama.cpp's penalty: an unknown character costs ten nats more than the
	// least likely piece the vocabulary has.
	t.unknownScore = minScore - 10
	sort.SliceStable(t.special, func(a, b int) bool {
		return len(pieces[t.special[a]]) > len(pieces[t.special[b]])
	})
	return t, nil
}

// Size is how many identifiers the vocabulary has.
func (t *Tokenizer) Size() int { return len(t.pieces) }

// Piece is the written form of an identifier, with the escaped space put back.
func (t *Tokenizer) Piece(id int32) string {
	if id < 0 || int(id) >= len(t.pieces) {
		return ""
	}
	if t.types[id] == typeNormal {
		return strings.ReplaceAll(t.pieces[id], escapedSpace, " ")
	}
	return t.pieces[id]
}

// Encode segments text. addSpecial frames it with the markers the file asks
// for — <s> before and </s> after, for this vocabulary — and parseSpecial says
// whether a marker's own text inside it is that marker or plain characters.
func (t *Tokenizer) Encode(text string, addSpecial, parseSpecial bool) []int32 {
	var out []int32
	if addSpecial && t.addBOS && t.BOS >= 0 {
		out = append(out, int32(t.BOS))
	}
	for _, f := range t.partition(text, parseSpecial) {
		if f.id >= 0 {
			out = append(out, int32(f.id))
			continue
		}
		out = t.segment(f.text, out)
	}
	if addSpecial && t.addEOS && t.EOS >= 0 {
		out = append(out, int32(t.EOS))
	}
	return out
}

// fragment is a stretch of text still to be segmented, or a marker already
// recognised as one.
type fragment struct {
	text string
	id   int
}

// partition splits the text on the special pieces, longest first, the way
// llama.cpp's tokenizer_st_partition does. A control or unknown piece is only
// looked for when the caller asked for special parsing; a user-defined one
// always is.
func (t *Tokenizer) partition(text string, parseSpecial bool) []fragment {
	frags := []fragment{{text: text, id: -1}}
	for _, id := range t.special {
		if !parseSpecial && t.types[id] != typeUserDefined {
			continue
		}
		piece := t.pieces[id]
		var next []fragment
		for _, f := range frags {
			if f.id >= 0 {
				next = append(next, f)
				continue
			}
			rest := f.text
			for {
				at := strings.Index(rest, piece)
				if at < 0 {
					break
				}
				if at > 0 {
					next = append(next, fragment{text: rest[:at], id: -1})
				}
				next = append(next, fragment{id: id})
				rest = rest[at+len(piece):]
			}
			if rest != "" {
				next = append(next, fragment{text: rest, id: -1})
			}
		}
		frags = next
	}
	return frags
}

// best is the best segmentation found so far of the text up to some offset:
// the last piece of it, where that piece starts, and the total score.
type best struct {
	id    int
	start int
	score float64
}

// segment is llm_tokenizer_ugm_session::tokenize: normalize, then walk the
// text a code point at a time, extending every piece the trie finds from each
// point, and read the best path back from the end.
func (t *Tokenizer) segment(text string, out []int32) []int32 {
	s := t.normalize(text)
	n := len(s)
	if n == 0 {
		return out
	}
	results := make([]best, n+1)
	for i := range results {
		results[i] = best{id: t.Unknown, score: -math.MaxFloat64}
	}
	results[0] = best{id: t.Unknown, score: 0}

	for at := 0; at < n; {
		width := min(utf8Len(s[at]), n-at)
		here := results[at].score
		whole := false
		node := 0
		for end := at; end < n; end++ {
			node = t.trie.child(node, s[end])
			if node < 0 {
				break
			}
			id := t.trie.value[node]
			if id < 0 {
				continue
			}
			if end+1-at == width {
				whole = true
			}
			// User-defined pieces score zero, so that they win: every normal
			// score is a log probability and negative. The sum is a double,
			// as in llama.cpp, which does it to match the HF tokenizer.
			score := float64(t.scores[id])
			if t.types[id] == typeUserDefined {
				score = 0
			}
			if c := here + score; c > results[end+1].score {
				results[end+1] = best{id: id, start: at, score: c}
			}
		}
		if !whole {
			if c := here + t.unknownScore; c > results[at+width].score {
				results[at+width] = best{id: t.Unknown, start: at, score: c}
			}
		}
		at += width
	}

	// Backwards, folding a run of unknowns into one.
	mark := len(out)
	prevUnknown := false
	for r := results[n]; ; r = results[r.start] {
		unknown := r.id == t.Unknown
		if !(prevUnknown && unknown) {
			out = append(out, int32(r.id))
		}
		if r.start == 0 {
			break
		}
		prevUnknown = unknown
	}
	for i, j := mark, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// utf8Len is llama.cpp's unicode_len_utf8: the length a lead byte announces,
// with a stray continuation byte counted as one.
func utf8Len(b byte) int {
	return [16]int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}[b>>4]
}

// normalize is llm_tokenizer_ugm_session::normalize. Each prefix of the input
// becomes what the map says, or itself; then the spaces are dealt with — the
// map has already made every kind of space an ordinary one — by prefixing one,
// merging runs, and escaping them all.
func (t *Tokenizer) normalize(text string) string {
	space := " "
	if t.escapeSpaces {
		space = escapedSpace
	}
	// treat_whitespace_as_suffix is never set for this vocabulary: llama.cpp
	// only reads it from a .model file, and a GGUF has no key for it.
	prepend := t.addSpacePrefix
	merge := t.removeExtraSpaces

	var b strings.Builder
	b.Grow(len(text) * 3)
	prepended, inWord := false, false
	for at := 0; at < len(text); {
		norm, consumed := t.normalizePrefix(text, at)
		for i := 0; i < len(norm); i++ {
			c := norm[i]
			if c != ' ' {
				if !inWord {
					inWord = true
					if (prepend && !prepended) || merge {
						b.WriteString(space)
						prepended = true
					}
				}
				b.WriteByte(c)
				continue
			}
			inWord = false
			if !merge {
				b.WriteString(space)
			}
		}
		at += consumed
	}
	return b.String()
}

// normalizePrefix is what the text at this offset becomes, and how many of its
// bytes that used up.
func (t *Tokenizer) normalizePrefix(text string, at int) (string, int) {
	if n := t.userDefined.longest(text[at:]); n > 0 {
		return text[at : at+n], n
	}
	if repl, n := t.charsmap.longest(text[at:]); n > 0 {
		return repl, n
	}
	r, size := utf8.DecodeRuneInString(text[at:])
	if r == utf8.RuneError && size <= 1 {
		// Not UTF-8: one byte goes, and U+FFFD takes its place.
		return "�", 1
	}
	return text[at : at+size], size
}

// floatArray reads a metadata array of floats.
func floatArray(g *tensors.GGUF, key string) ([]float32, error) {
	raw, ok := g.Meta[key]
	if !ok {
		return nil, fmt.Errorf("metadata %q is absent", key)
	}
	items, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("metadata %q is %T, not an array", key, raw)
	}
	out := make([]float32, len(items))
	for i, v := range items {
		f, ok := v.(float32)
		if !ok {
			return nil, fmt.Errorf("metadata %q holds a %T, not a float", key, v)
		}
		out[i] = f
	}
	return out, nil
}

// byteArray reads the character map, which the converter writes as an array
// of eight-bit integers.
func byteArray(raw any) ([]byte, error) {
	items, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("ugm: the character map is %T, not an array", raw)
	}
	out := make([]byte, len(items))
	for i, v := range items {
		switch b := v.(type) {
		case uint8:
			out[i] = b
		case int8:
			out[i] = byte(b)
		default:
			return nil, fmt.Errorf("ugm: the character map holds a %T", v)
		}
	}
	return out, nil
}
