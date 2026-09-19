package krea2

import (
	"strings"

	"github.com/ThiraSoft/golem/token/bytebpe"
)

// promptTemplate is KREA2_TEMPLATE (comfy/text_encoders/krea2.py): Qwen-Image's
// system prompt around the user's text, with the assistant's turn opened and
// no thinking block.
const promptTemplate = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"

const (
	imStart   = 151644
	userToken = 872
	newline   = 198
)

// Prompt is a prompt as the text encoder reads it: every token of the
// template and the text, and where the conditioning starts once the
// template's prefix is cut away.
type Prompt struct {
	IDs   []int32
	Start int
}

// Tokenize does what ComfyUI's Krea2Tokenizer does with a prompt: the
// template around it, tokenized whole with the control tokens recognized,
// and the prefix up to the user's text found the way
// Krea2TEModel.encode_token_weights finds it.
//
// Krea 2's tokenizer does not weigh: "(red:1.3)" is text, brackets and all.
// The one thing ComfyUI's bracket syntax still does is unescape, so "\(" is a
// bracket.
func Tokenize(v *bytebpe.Vocab, text string) Prompt {
	text = strings.NewReplacer(`\(`, "(", `\)`, ")").Replace(text)
	full := strings.Replace(promptTemplate, "%s", text, 1)
	p := Prompt{IDs: v.Encode(full, false, true)}
	// The second <|im_start|> opens the user's turn; "user\n" after it is
	// template too.
	end, seen := -1, 0
	for i, id := range p.IDs {
		if id == imStart && seen < 2 {
			end = i
			seen++
		}
	}
	if len(p.IDs) > end+3 && p.IDs[end+1] == userToken && p.IDs[end+2] == newline {
		end += 3
	}
	p.Start = end
	return p
}
