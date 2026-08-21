package bytebpe

// Turning identifiers back into text.
//
// A normal piece gives its bytes back through the alphabet. A control piece
// gives nothing at all unless the caller asks to see it — which is what makes
// a transcript readable and a debug dump complete.

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
		// Written as it stands: these are the markers a template writes, and
		// they were never put through the alphabet.
		return text
	default:
		bytes, err := decodeBytes(text)
		if err != nil {
			// A normal piece that is not in the alphabet cannot be turned into
			// text at all. Load checks the whole vocabulary, so reaching here
			// means the file changed under us.
			return ""
		}
		return string(bytes)
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
