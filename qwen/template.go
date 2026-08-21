package qwen

// The chat.Template this engine implements.

import "github.com/ThiraSoft/golem/chat"

// Template holds the config so it has the same shape as Gemma's, which carries
// a checkpoint quirk in it. Qwen's template has no such quirk today.
type Template struct{ cfg *Config }

func NewTemplate(cfg *Config) *Template { return &Template{cfg: cfg} }

func (t *Template) Render(msgs []chat.Message, opt chat.Options) (string, error) {
	return RenderChat(msgs, opt)
}

func (t *Template) ParseCalls(text string) (string, []chat.ToolCall, error) {
	return ParseToolCalls(text)
}

func (t *Template) CallOpen() string { return toolCallOpen }
