package bytebpe

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/ThiraSoft/golem/token/merge"
	"github.com/ThiraSoft/golem/token/special"
)

// LoadTokenizerJSON reads a byte-level BPE out of the tokenizer.json Hugging
// Face's fast tokenizers write, which is how an encoder published as
// safetensors carries its vocabulary.
//
// Only the shape ModernBERT uses is accepted: a BPE model, a ByteLevel
// pre-tokenizer with its regex and no prefix space, and either no normalizer
// or NFC. NFC is not applied, as LoadFiles does not apply it: a text holding
// decomposed accents is cut differently, and nothing else is. Anything else is
// refused rather than tokenized by rules the file did not ask for.
//
// Every added token is cut out of the text before the rest is split, special or
// not, which is what the reference does when a text is encoded without its
// special tokens added: they are still recognized.
func LoadTokenizerJSON(path string) (*Vocab, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file struct {
		Normalizer *struct {
			Type string `json:"type"`
		} `json:"normalizer"`
		PreTokenizer *struct {
			Type           string `json:"type"`
			AddPrefixSpace bool   `json:"add_prefix_space"`
			UseRegex       bool   `json:"use_regex"`
		} `json:"pre_tokenizer"`
		Model struct {
			Type         string            `json:"type"`
			Vocab        map[string]int32  `json:"vocab"`
			Merges       []json.RawMessage `json:"merges"`
			ByteFallback bool              `json:"byte_fallback"`
		} `json:"model"`
		Added []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if file.Model.Type != "BPE" || file.Model.ByteFallback {
		return nil, fmt.Errorf("%s: model %q (byte fallback %v) is not a byte-level BPE", path, file.Model.Type, file.Model.ByteFallback)
	}
	if p := file.PreTokenizer; p == nil || p.Type != "ByteLevel" || p.AddPrefixSpace || !p.UseRegex {
		return nil, fmt.Errorf("%s: the pre-tokenizer is not ByteLevel with GPT-2's regex and no prefix space", path)
	}
	if n := file.Normalizer; n != nil && n.Type != "NFC" {
		return nil, fmt.Errorf("%s: normalizer %q is not implemented", path, n.Type)
	}

	size := int32(0)
	for _, id := range file.Model.Vocab {
		size = max(size, id+1)
	}
	for _, a := range file.Added {
		size = max(size, a.ID+1)
	}
	v := &Vocab{
		texts: make([]string, size),
		kinds: make([]Kind, size),
		index: make(map[string]int32, size),
		ranks: make(map[merge.Pair]int, len(file.Model.Merges)),
		eog:   map[int32]bool{},
		bos:   -1,
		eos:   -1,
		split: splitGPT2,
	}
	for text, id := range file.Model.Vocab {
		v.texts[id], v.kinds[id], v.index[text] = text, Normal, id
	}
	for _, a := range file.Added {
		kind := UserDefined
		if a.Special {
			kind = Control
		}
		v.texts[a.ID], v.kinds[a.ID], v.index[a.Content] = a.Content, kind, a.ID
		// Not hidden: the reference recognizes the special ones in text too.
		v.specials = append(v.specials, special.Token{ID: a.ID, Text: a.Content})
	}
	for id, k := range v.kinds {
		if k == 0 {
			v.kinds[id] = Unused
		}
	}
	special.Sort(v.specials)

	// A merge is written "left right" by older files and ["left", "right"]
	// by newer ones.
	for rank, m := range file.Model.Merges {
		var pair [2]string
		var line string
		if err := json.Unmarshal(m, &line); err == nil {
			left, right, found := strings.Cut(line, " ")
			if !found || strings.ContainsRune(right, ' ') {
				return nil, fmt.Errorf("%s: merge %d %q is not two pieces", path, rank, line)
			}
			pair = [2]string{left, right}
		} else if err := json.Unmarshal(m, &pair); err != nil {
			return nil, fmt.Errorf("%s: merge %d: %w", path, rank, err)
		}
		v.ranks[merge.Pair{Left: pair[0], Right: pair[1]}] = rank
	}
	return v, nil
}
