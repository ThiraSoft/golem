package yue2

import (
	"fmt"

	"github.com/ThiraSoft/golem/tensors"
	"github.com/ThiraSoft/golem/token/bytebpe"
	"golang.org/x/text/unicode/norm"
)

// Tokenizer wraps a byte-level BPE vocabulary loaded from the YuE2 GGUF file.
// Text is normalized to Unicode NFC before encoding. Special tokens are never
// recognized inside the text and are positioned by id by the prompt assembler.
type Tokenizer struct {
	v *bytebpe.Vocab
}

// NewTokenizer loads the GPT-2 byte BPE vocabulary and merges from the given GGUF.
func NewTokenizer(g *tensors.GGUF) (*Tokenizer, error) {
	v, err := bytebpe.Load(g, bytebpe.WithDefaultPre("qwen2"))
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	return &Tokenizer{v: v}, nil
}

// Encode converts input text to token IDs after Unicode NFC normalization.
// Special tokens are never matched inside text.
func (t *Tokenizer) Encode(text string) []int32 {
	nfcText := norm.NFC.String(text)
	return t.v.Encode(nfcText, false, false)
}

// Decode reconstructs text from a sequence of token IDs.
func (t *Tokenizer) Decode(ids []int32) string {
	return t.v.Decode(ids, false)
}
