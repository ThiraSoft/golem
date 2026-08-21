// Package bpe reads Gemma 4's tokenizer out of a GGUF and applies it.
//
// The file declares tokenizer.ggml.model = "gemma4", which llama.cpp treats as
// BPE over raw UTF-8 with SentencePiece's whitespace escaping: spaces become
// U+2581 before the merges run, and there is no GPT-2 byte alphabet. Nothing
// here is shared with token/sentencepiece beyond that character.
package bpe

import (
	"fmt"
	"strings"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/merge"
	"github.com/ThiraSoft/golem/token/special"
)

// Space is the character standing in for the space, so that the pieces keep
// track of word boundaries.
const Space = "▁"

// Kind is what the file says a piece is. The numbering is the format's.
type Kind uint32

const (
	Normal      Kind = 1
	Unknown     Kind = 2
	Control     Kind = 3
	UserDefined Kind = 4
	Unused      Kind = 5
	Byte        Kind = 6
)

// Vocab is a loaded tokenizer: the pieces, what each one is, and the ranked
// merges that build them.
type Vocab struct {
	texts    []string
	kinds    []Kind
	index    map[string]int32
	ranks    map[merge.Pair]int
	specials []special.Token // control, unknown and user-defined, longest text first
	byteIDs  [256]int32

	bos, eos, unk int32
	addBOS        bool
}

// Load reads the tokenizer from an already-open GGUF.
func Load(g *tensors.GGUF) (*Vocab, error) {
	texts, err := g.Strings("tokenizer.ggml.tokens")
	if err != nil {
		return nil, err
	}
	rawKinds, err := g.Uint32Slice("tokenizer.ggml.token_type")
	if err != nil {
		return nil, err
	}
	if len(rawKinds) != len(texts) {
		return nil, fmt.Errorf("%d tokens but %d types", len(texts), len(rawKinds))
	}
	merges, err := g.Strings("tokenizer.ggml.merges")
	if err != nil {
		return nil, err
	}

	v := &Vocab{
		texts: texts,
		kinds: make([]Kind, len(rawKinds)),
		index: make(map[string]int32, len(texts)),
		ranks: make(map[merge.Pair]int, len(merges)),
	}
	for i := range v.byteIDs {
		v.byteIDs[i] = -1
	}
	for i, k := range rawKinds {
		v.kinds[i] = Kind(k)
	}
	for id, text := range texts {
		// First one wins; the vocabulary has no duplicates, but better not to
		// depend on that.
		if _, seen := v.index[text]; !seen {
			v.index[text] = int32(id)
		}
		switch v.kinds[id] {
		case Byte:
			if b, ok := byteValue(text); ok {
				v.byteIDs[b] = int32(id)
			}
		case Control, Unknown, UserDefined:
			v.specials = append(v.specials, special.Token{
				ID:     int32(id),
				Text:   text,
				Hidden: v.kinds[id] != UserDefined,
			})
		}
	}

	for rank, m := range merges {
		// The separator is a space, and the left half may itself start with
		// one — the first entries of the table are runs of U+2581 and of
		// newlines. The search therefore starts at index 1.
		cut := strings.Index(m[1:], " ")
		if cut < 0 {
			return nil, fmt.Errorf("merge %d has no separator: %q", rank, m)
		}
		cut++
		v.ranks[merge.Pair{Left: m[:cut], Right: m[cut+1:]}] = rank
	}

	// Longest first: a short special token may be a substring of a longer one,
	// and taking it first would cut the longer one in half.
	special.Sort(v.specials)

	v.bos = metaID(g, "tokenizer.ggml.bos_token_id")
	v.eos = metaID(g, "tokenizer.ggml.eos_token_id")
	v.unk = metaID(g, "tokenizer.ggml.unknown_token_id")
	if add, err := g.Bool("tokenizer.ggml.add_bos_token"); err == nil {
		v.addBOS = add
	} else {
		// llama.cpp forces it on for gemma4 whatever the file says.
		v.addBOS = true
	}
	return v, nil
}

// metaID reads an optional identifier; a missing one is -1 rather than an
// error, because not every checkpoint declares every special token.
func metaID(g *tensors.GGUF, key string) int32 {
	value, err := g.Uint32(key)
	if err != nil {
		return -1
	}
	return int32(value)
}

// byteValue recognizes the "<0x41>" pieces.
func byteValue(s string) (byte, bool) {
	if len(s) != 6 || !strings.HasPrefix(s, "<0x") || s[5] != '>' {
		return 0, false
	}
	var v byte
	for _, c := range []byte(s[3:5]) {
		switch {
		case c >= '0' && c <= '9':
			v = v<<4 | (c - '0')
		case c >= 'A' && c <= 'F':
			v = v<<4 | (c - 'A' + 10)
		case c >= 'a' && c <= 'f':
			v = v<<4 | (c - 'a' + 10)
		default:
			return 0, false
		}
	}
	return v, true
}

// Size is the number of pieces.
func (v *Vocab) Size() int { return len(v.texts) }

// Kind reports what the file says a piece is.
func (v *Vocab) Kind(id int32) Kind {
	if id < 0 || int(id) >= len(v.kinds) {
		return Unused
	}
	return v.kinds[id]
}

// Text is the written form of a piece, escaping and all.
func (v *Vocab) Text(id int32) string {
	if id < 0 || int(id) >= len(v.texts) {
		return ""
	}
	return v.texts[id]
}

// ID looks a piece up by its written form.
func (v *Vocab) ID(text string) (int32, bool) {
	id, ok := v.index[text]
	return id, ok
}

func (v *Vocab) BOS() int32   { return v.bos }
func (v *Vocab) EOS() int32   { return v.eos }
func (v *Vocab) AddBOS() bool { return v.addBOS }

func (v *Vocab) specialsByLength() []special.Token { return v.specials }
