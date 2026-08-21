package gemma

// The chat.Template this engine implements. The rendering itself lives in
// chat.go and does not change; what changes is that the caller no longer has
// to know which of the two checkpoints it is holding.

import "github.com/ThiraSoft/golem/chat"

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
