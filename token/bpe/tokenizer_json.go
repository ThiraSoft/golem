package bpe

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/ThiraSoft/golem/token/merge"
	"github.com/ThiraSoft/golem/token/special"
)

// LoadTokenizerJSON reads a SentencePiece-style BPE out of the tokenizer.json
// Hugging Face's fast tokenizers write: the shape mmBERT's tokenizer has, and
// with it the multilingual Laya.
//
// It differs from what Load reads in how the text is cut before the merges.
// The normalizer turns every space into U+2581; the Metaspace pre-tokenizer
// puts one in front of the text when it does not start with one, and cuts the
// text before every U+2581, each word keeping its own. So a merge never
// straddles two words, where gemma4's straddles everything but a newline.
//
// Only that shape is accepted: a BPE model with byte fallback and no merge
// skipping, a Replace of " " by U+2581 as the normalizer, Metaspace with
// prepend_scheme "always" and split on. Every added token is cut out of the
// text first, as the reference does when it encodes without adding special
// tokens: they are still recognized.
func LoadTokenizerJSON(path string) (*Vocab, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file struct {
		Normalizer *struct {
			Type    string `json:"type"`
			Pattern struct {
				String string `json:"String"`
			} `json:"pattern"`
			Content string `json:"content"`
		} `json:"normalizer"`
		PreTokenizer *struct {
			Type        string `json:"type"`
			Replacement string `json:"replacement"`
			Prepend     string `json:"prepend_scheme"`
			Split       bool   `json:"split"`
		} `json:"pre_tokenizer"`
		Model struct {
			Type         string            `json:"type"`
			Vocab        map[string]int32  `json:"vocab"`
			Merges       []json.RawMessage `json:"merges"`
			ByteFallback bool              `json:"byte_fallback"`
			IgnoreMerges bool              `json:"ignore_merges"`
			Unk          string            `json:"unk_token"`
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
	if file.Model.Type != "BPE" || !file.Model.ByteFallback || file.Model.IgnoreMerges {
		return nil, fmt.Errorf("%s: model %q is not a BPE with byte fallback", path, file.Model.Type)
	}
	if n := file.Normalizer; n == nil || n.Type != "Replace" || n.Pattern.String != " " || n.Content != Space {
		return nil, fmt.Errorf("%s: the normalizer is not a Replace of spaces by U+2581", path)
	}
	if p := file.PreTokenizer; p == nil || p.Type != "Metaspace" || p.Replacement != Space || p.Prepend != "always" || !p.Split {
		return nil, fmt.Errorf("%s: the pre-tokenizer is not Metaspace, always prepending and splitting", path)
	}

	size := int32(0)
	for _, id := range file.Model.Vocab {
		size = max(size, id+1)
	}
	for _, a := range file.Added {
		size = max(size, a.ID+1)
	}
	v := &Vocab{
		texts:     make([]string, size),
		kinds:     make([]Kind, size),
		index:     make(map[string]int32, size),
		ranks:     make(map[merge.Pair]int, len(file.Model.Merges)),
		bos:       -1,
		eos:       -1,
		unk:       -1,
		metaspace: true,
	}
	for i := range v.byteIDs {
		v.byteIDs[i] = -1
	}
	for text, id := range file.Model.Vocab {
		v.texts[id], v.kinds[id], v.index[text] = text, Normal, id
		if b, ok := byteValue(text); ok {
			v.kinds[id] = Byte
			v.byteIDs[b] = id
		}
	}
	for _, a := range file.Added {
		kind := UserDefined
		if a.Special {
			kind = Control
		}
		v.texts[a.ID], v.kinds[a.ID], v.index[a.Content] = a.Content, kind, a.ID
		v.specials = append(v.specials, special.Token{ID: a.ID, Text: a.Content})
	}
	for id, k := range v.kinds {
		if k == 0 {
			v.kinds[id] = Unused
		}
	}
	special.Sort(v.specials)
	if id, ok := v.index[file.Model.Unk]; ok {
		v.unk = id
	}

	for rank, m := range file.Model.Merges {
		var pair [2]string
		var line string
		if err := json.Unmarshal(m, &line); err == nil {
			// The older form; a half may itself start with a space here, as
			// in Load's table, so the search starts at index 1.
			cut := strings.Index(line[1:], " ")
			if cut < 0 {
				return nil, fmt.Errorf("%s: merge %d %q has no separator", path, rank, line)
			}
			pair = [2]string{line[:cut+1], line[cut+2:]}
		} else if err := json.Unmarshal(m, &pair); err != nil {
			return nil, fmt.Errorf("%s: merge %d: %w", path, rank, err)
		}
		// The first rank wins, as it does in the reference's table.
		key := merge.Pair{Left: pair[0], Right: pair[1]}
		if _, seen := v.ranks[key]; !seen {
			v.ranks[key] = rank
		}
	}
	return v, nil
}

// splitMetaspace cuts escaped text before every U+2581, each word keeping the
// one that starts it, after putting one in front when there is none.
func splitMetaspace(text string) []string {
	if !strings.HasPrefix(text, Space) {
		text = Space + text
	}
	var words []string
	for len(text) > 0 {
		next := strings.Index(text[len(Space):], Space)
		if next < 0 {
			words = append(words, text)
			break
		}
		words = append(words, text[:next+len(Space)])
		text = text[next+len(Space):]
	}
	return words
}
