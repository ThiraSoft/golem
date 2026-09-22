package gemma

// The chat.Template this engine implements. The rendering itself lives in
// chat.go and does not change; what changes is that the caller no longer has
// to know which of the two checkpoints it is holding.

import (
	"strings"

	"github.com/ThiraSoft/golem/chat"
)

// FileMedia turns the one token the file's template writes for a picture or a
// recording into the empty pair of markers RenderChat writes, which is what
// BuildPrompt fills in.
var FileMedia = strings.NewReplacer(imageSoft, imageOpen+imageClose, audioSoft, audioOpen+audioClose)

type Template struct{ cfg *Config }

func NewTemplate(cfg *Config) *Template { return &Template{cfg: cfg} }

func (t *Template) Render(msgs []chat.Message, opt chat.Options) (string, error) {
	return RenderChat(msgs, ChatOptions{
		EnableThinking:      opt.EnableThinking,
		EmptyThought:        t.cfg.EmptyThought,
		Tools:               opt.Tools,
		AddGenerationPrompt: opt.AddGenerationPrompt,
	})
}

func (t *Template) ParseCalls(text string) (string, []chat.ToolCall, error) {
	return ParseToolCalls(text)
}

func (t *Template) CallOpen() string { return toolCallOpen }

// ReasoningMarkers are the thought channel's. The channel's name is part of
// the opening marker: thought is the only channel the checkpoints write.
func (t *Template) ReasoningMarkers() (string, string) {
	return channelOpen + "thought", channelClose
}
