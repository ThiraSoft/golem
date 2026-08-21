// Package bytebpe reads a byte-level BPE tokenizer out of a GGUF and applies
// it — the kind GPT-2 introduced and Qwen uses.
//
// The file declares tokenizer.ggml.model = "gpt2". Two things follow, and they
// are the two things token/bpe does not do. Every byte is given a printable
// stand-in before anything else happens, so the vocabulary is a list of strings
// that can hold arbitrary bytes and a space is written U+0120. And the text is
// cut into words before any merge runs, by a set of rules the file names in
// tokenizer.ggml.pre — "qwen2" here.
//
// The merging between those two is not this package's: it is token/merge, which
// llama.cpp does the same way for both tokenizers.
package bytebpe

import (
	"fmt"
	"strings"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/merge"
	"github.com/ThiraSoft/golem/token/special"
)

// Kind is what the file says a piece is. The numbering is the format's, the
// same one token/bpe reads.
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
	eog      map[int32]bool

	eos    int32
	addBOS bool
	bos    int32
}

// Load reads the tokenizer from an already-open GGUF.
func Load(g *tensors.GGUF) (*Vocab, error) {
	model, err := g.String("tokenizer.ggml.model")
	if err != nil {
		return nil, err
	}
	if model != "gpt2" {
		return nil, fmt.Errorf("tokenizer model %q is not gpt2", model)
	}
	pre, err := g.String("tokenizer.ggml.pre")
	if err != nil {
		return nil, err
	}
	if !knownPre(pre) {
		return nil, fmt.Errorf("pre-tokenizer %q is not one this package implements", pre)
	}

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
		eog:   make(map[int32]bool, 4),
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
		case Control, Unknown, UserDefined:
			v.specials = append(v.specials, special.Token{
				ID:     int32(id),
				Text:   text,
				Hidden: v.kinds[id] != UserDefined,
			})
		}
	}

	for rank, m := range merges {
		// Unlike gemma4's table, neither half can contain a space: every byte
		// has been replaced by a printable stand-in, and U+0020 is not one of
		// them. So the separator is the only space on the line, and finding it
		// needs no care.
		left, right, found := strings.Cut(m, " ")
		if !found {
			return nil, fmt.Errorf("merge %d has no separator: %q", rank, m)
		}
		if strings.ContainsRune(right, ' ') {
			return nil, fmt.Errorf("merge %d has two separators: %q", rank, m)
		}
		v.ranks[merge.Pair{Left: left, Right: right}] = rank
	}

	// Longest first: a short special token may be a substring of a longer one,
	// and taking it first would cut the longer one in half.
	special.Sort(v.specials)

	v.bos = metaID(g, "tokenizer.ggml.bos_token_id")
	v.eos = metaID(g, "tokenizer.ggml.eos_token_id")
	if add, err := g.Bool("tokenizer.ggml.add_bos_token"); err == nil {
		v.addBOS = add
	}

	// What ends a generation. The declared end-of-sentence, and the two markers
	// this family of templates closes a turn with — read by name rather than by
	// number, because the numbers are this checkpoint's and the names are the
	// format's.
	if v.eos >= 0 {
		v.eog[v.eos] = true
	}
	for _, name := range []string{"<|im_end|>", "<|endoftext|>"} {
		if id, ok := v.index[name]; ok {
			v.eog[id] = true
		}
	}
	return v, nil
}

// knownPre reports whether the pre-tokenizer named in the file is one this
// package splits text for. A name it does not know is refused at load rather
// than silently tokenized by the wrong rules, which would cost a handful of
// tokens per sentence and never raise an error.
func knownPre(pre string) bool {
	switch pre {
	case "qwen2", "deepseek-r1-qwen":
		return true
	}
	return false
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

// Size is the number of pieces.
func (v *Vocab) Size() int { return len(v.texts) }

// Kind reports what the file says a piece is.
func (v *Vocab) Kind(id int32) Kind {
	if id < 0 || int(id) >= len(v.kinds) {
		return Unused
	}
	return v.kinds[id]
}

// Text is the written form of a piece, in the byte alphabet: the stand-ins are
// what the file holds, and Piece is what turns them back into bytes.
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

// IsEOG reports whether the identifier ends a generation.
func (v *Vocab) IsEOG(id int32) bool { return v.eog[id] }

func (v *Vocab) BOS() int32   { return v.bos }
func (v *Vocab) EOS() int32   { return v.eos }
func (v *Vocab) AddBOS() bool { return v.addBOS }
