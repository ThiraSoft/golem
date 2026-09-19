package krea2

import (
	"strconv"
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
// template and the text, the weight ComfyUI's bracket syntax gave each, and
// where the conditioning starts once the template's prefix is cut away.
type Prompt struct {
	IDs     []int32
	Weights []float32
	Start   int
}

// Weighted reports whether any token's weight is not 1.
func (p Prompt) Weighted() bool {
	for _, w := range p.Weights {
		if w != 1 {
			return true
		}
	}
	return false
}

// Tokenize does what ComfyUI's Krea2Tokenizer does with a prompt: the
// template around it, the bracket weights parsed out of the whole ("(word)" is
// ×1.1, "(word:1.3)" is 1.3, "\(" is a bracket), each weighted piece tokenized
// on its own with the control tokens recognized, and the prefix up to the
// user's text found the way Krea2TEModel.encode_token_weights finds it.
func Tokenize(v *bytebpe.Vocab, text string) Prompt {
	full := strings.Replace(promptTemplate, "%s", text, 1)
	var p Prompt
	for _, seg := range tokenWeights(escapeImportant(full), 1) {
		s := unescapeImportant(seg.text)
		if s == "" {
			continue
		}
		for _, id := range v.Encode(s, false, true) {
			p.IDs = append(p.IDs, id)
			p.Weights = append(p.Weights, seg.weight)
		}
	}
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

type segment struct {
	text   string
	weight float32
}

func escapeImportant(s string) string {
	return strings.NewReplacer(`\)`, "\x00\x01", `\(`, "\x00\x02").Replace(s)
}

func unescapeImportant(s string) string {
	return strings.NewReplacer("\x00\x01", ")", "\x00\x02", "(").Replace(s)
}

// parseParentheses cuts s at its outermost brackets, keeping them.
func parseParentheses(s string) []string {
	var out []string
	var cur strings.Builder
	level := 0
	for _, r := range s {
		switch {
		case r == '(':
			if level == 0 {
				if cur.Len() > 0 {
					out = append(out, cur.String())
				}
				cur.Reset()
			}
			cur.WriteRune(r)
			level++
		case r == ')':
			level--
			if level == 0 {
				cur.WriteRune(r)
				out = append(out, cur.String())
				cur.Reset()
			} else {
				cur.WriteRune(r)
			}
		default:
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

// tokenWeights is sd1_clip.token_weights. The arithmetic is Python's, in
// double precision, and the weight lands in the token list as a float.
func tokenWeights(s string, weight float64) []segment {
	var out []segment
	for _, x := range parseParentheses(s) {
		if len(x) >= 2 && x[0] == '(' && x[len(x)-1] == ')' {
			x = x[1 : len(x)-1]
			w := weight * 1.1
			if at := strings.LastIndex(x, ":"); at > 0 {
				if f, err := strconv.ParseFloat(strings.TrimSpace(x[at+1:]), 64); err == nil {
					w = f
					x = x[:at]
				}
			}
			out = append(out, tokenWeights(x, w)...)
		} else {
			out = append(out, segment{x, float32(weight)})
		}
	}
	return out
}
