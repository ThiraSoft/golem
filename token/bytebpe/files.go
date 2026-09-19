package bytebpe

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ThiraSoft/golem/token/merge"
	"github.com/ThiraSoft/golem/token/special"
)

// LoadFiles reads the same tokenizer from the files Hugging Face's
// Qwen2Tokenizer is built from, as ComfyUI ships them: vocab.json,
// merges.txt, and tokenizer_config.json for the tokens added after the
// merges. It is how a checkpoint that carries no vocabulary of its own, such
// as a text encoder saved as safetensors, is tokenized.
//
// The pre-tokenizer is qwen2's, which is the one that file format implies.
// Hugging Face puts the text through Unicode NFC first; this does not, so a
// prompt holding decomposed accents is cut differently. ASCII is not affected.
func LoadFiles(dir string) (*Vocab, error) {
	var pieces map[string]int32
	if err := readJSON(filepath.Join(dir, "vocab.json"), &pieces); err != nil {
		return nil, err
	}
	var config struct {
		Added map[string]struct {
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens_decoder"`
		AddBOS bool   `json:"add_bos_token"`
		EOS    string `json:"eos_token"`
	}
	if err := readJSON(filepath.Join(dir, "tokenizer_config.json"), &config); err != nil {
		return nil, err
	}

	size := int32(len(pieces))
	for _, id := range pieces {
		size = max(size, id+1)
	}
	added := make(map[int32]Kind, len(config.Added))
	for key, tok := range config.Added {
		var id int32
		if _, err := fmt.Sscan(key, &id); err != nil {
			return nil, fmt.Errorf("added token %q: %w", key, err)
		}
		size = max(size, id+1)
		pieces[tok.Content] = id
		added[id] = UserDefined
		if tok.Special {
			added[id] = Control
		}
	}

	v := &Vocab{
		texts: make([]string, size),
		kinds: make([]Kind, size),
		index: make(map[string]int32, size),
		eog:   make(map[int32]bool, 4),
		bos:   -1,
		eos:   -1,
	}
	for text, id := range pieces {
		v.texts[id] = text
		v.kinds[id] = Normal
		v.index[text] = id
	}
	for id := range v.kinds {
		if v.kinds[id] == 0 {
			return nil, fmt.Errorf("%s: no piece has identifier %d", dir, id)
		}
		if k, ok := added[int32(id)]; ok {
			v.kinds[id] = k
			v.specials = append(v.specials, special.Token{ID: int32(id), Text: v.texts[id], Hidden: k != UserDefined})
		}
	}
	special.Sort(v.specials)

	f, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	v.ranks = make(map[merge.Pair]int, 151387)
	scan := bufio.NewScanner(f)
	for rank := 0; scan.Scan(); {
		line := scan.Text()
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		left, right, found := strings.Cut(line, " ")
		if !found || strings.ContainsRune(right, ' ') {
			return nil, fmt.Errorf("merges.txt: %q is not two pieces", line)
		}
		v.ranks[merge.Pair{Left: left, Right: right}] = rank
		rank++
	}
	if err := scan.Err(); err != nil {
		return nil, err
	}

	v.addBOS = config.AddBOS
	if id, ok := v.index[config.EOS]; ok {
		v.eos = id
		v.eog[id] = true
	}
	for _, name := range []string{"<|im_end|>", "<|endoftext|>"} {
		if id, ok := v.index[name]; ok {
			v.eog[id] = true
		}
	}
	return v, nil
}

func readJSON(path string, into any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
