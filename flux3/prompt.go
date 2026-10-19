package flux3

import (
	"strings"

	"github.com/ThiraSoft/golem/krea2"
)

// promptTemplate is Qwen3-VL's instruct chat template without a think block
// (comfy/text_encoders/flux.py Flux3Tokenizer).
const promptTemplate = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"

const (
	padToken = 151643
	padBlock = 80
)

// Tokenize wraps prompt in the Qwen3-VL instruct template, tokenizes it with
// tok, and right-pads it with padToken (151643) to a multiple of 80 tokens.
// real is the count of tokens before padding.
func Tokenize(tok *krea2.Tokenizer, prompt string) ([]int32, int) {
	text := strings.NewReplacer(`\(`, "(", `\)`, ")").Replace(prompt)
	full := strings.Replace(promptTemplate, "%s", text, 1)
	ids := tok.Encode(full, false, true)
	real := len(ids)
	pad := (padBlock - (real % padBlock)) % padBlock
	for i := 0; i < pad; i++ {
		ids = append(ids, padToken)
	}
	return ids, real
}
