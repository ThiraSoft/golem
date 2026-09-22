package qwen35

import (
	"strings"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/qwen"
)

// FileMedia empties the vision pair the file's template writes for a picture,
// the way this package's Template writes it: the pad between the markers is
// BuildPrompt's to fill, as many rows as the tower returns.
var FileMedia = strings.NewReplacer(VisionStart+ImagePad+VisionEnd, VisionStart+VisionEnd)

// Template writes a conversation the way this checkpoint reads one.
//
// It is Qwen3's, with one thing added: a turn carrying pictures opens each of
// them with an empty VisionStart/VisionEnd pair. BuildPrompt fills the pairs
// in — the template cannot, because how many rows a picture becomes is the
// tower's answer and is not known when the conversation is rendered.
type Template struct{ inner chat.Template }

// NewTemplate is this checkpoint's template.
func NewTemplate() *Template { return &Template{inner: qwen.NewTemplate(&qwen.Config{})} }

func (t *Template) Render(msgs []chat.Message, opt chat.Options) (string, error) {
	// The markers go in front of the turn's own text, which is where the
	// reference puts them: <|vision_start|><|vision_end|> then the question.
	out := make([]chat.Message, len(msgs))
	copy(out, msgs)
	for i := range out {
		if n := len(out[i].Images); n > 0 {
			var b strings.Builder
			for j := 0; j < n; j++ {
				b.WriteString(VisionStart)
				b.WriteString(VisionEnd)
			}
			b.WriteString(out[i].Content)
			out[i].Content = b.String()
		}
	}
	return t.inner.Render(out, opt)
}

func (t *Template) ParseCalls(text string) (string, []chat.ToolCall, error) {
	return t.inner.ParseCalls(text)
}

func (t *Template) CallOpen() string { return t.inner.CallOpen() }

func (t *Template) ReasoningMarkers() (string, string) { return t.inner.ReasoningMarkers() }
