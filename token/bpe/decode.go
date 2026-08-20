package bpe

// Turning identifiers back into text.
//
// Each kind of piece answers differently: a normal one gives up its U+2581
// again, a byte piece gives one byte, and a control piece gives nothing at all
// unless the caller asks to see it — which is what makes a transcript readable
// and a debug dump complete.

import "strings"

// Piece is the text of one identifier. special decides whether the control and
// unknown pieces are rendered or dropped.
func (v *Vocab) Piece(id int32, special bool) string {
	text := v.Text(id)
	if text == "" {
		return ""
	}
	switch v.Kind(id) {
	case Control, Unknown:
		if !special {
			return ""
		}
		return text
	case UserDefined:
		return text
	case Byte:
		if b, ok := byteValue(text); ok {
			return string([]byte{b})
		}
		return ""
	default:
		return strings.ReplaceAll(text, Space, " ")
	}
}

// Decode reconstructs the text of a sequence.
func (v *Vocab) Decode(ids []int32, special bool) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(v.Piece(id, special))
	}
	return b.String()
}

// IsEOG reports whether an identifier ends the model's turn. Gemma 4 has three
// of them and the file names none: they are recognized by their text, exactly
// as llama.cpp recognizes them.
func (v *Vocab) IsEOG(id int32) bool {
	switch v.Text(id) {
	case "<eos>", "<turn|>", "<|tool_response>":
		return true
	}
	return id == v.eos
}
