package chat

// The template the model file carries.
//
// A GGUF holds its chat template as Jinja under tokenizer.chat_template, and
// that template is the checkpoint's own word on how a conversation is spelled.
// FileTemplate renders with it, through golem/jinja, in the variables
// transformers hands it: the messages as OpenAI spells them, the tools, the
// two flags, and the special tokens.
//
// The engine's own template stays beside it. It parses the calls the model
// writes, which no chat template does, and it is what an engine falls back to
// when a file carries no template or one golem/jinja cannot read.

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ThiraSoft/golem/jinja"
)

type FileTemplate struct {
	tpl *jinja.Template
	own Template
	// tokens are the special tokens a template may print: bos_token and
	// eos_token.
	tokens map[string]any
	// media turns the placeholder the template writes for a picture or a
	// recording into the empty pair of markers the engine fills in. A
	// template writes one token where the tower's rows will go; the engine
	// wants the boundaries, and counts the rows itself.
	media *strings.Replacer
	// prepare adjusts the messages before they reach the template, for the
	// rare rule an engine keeps whatever the template says.
	prepare func([]Message) []Message
}

// NewFileTemplate parses src. own is the engine's template, which parses the
// calls and says where they open. tokens and media may be nil.
func NewFileTemplate(src string, own Template, tokens map[string]string, media *strings.Replacer) (*FileTemplate, error) {
	tpl, err := jinja.Parse(src)
	if err != nil {
		return nil, err
	}
	t := &FileTemplate{tpl: tpl, own: own, tokens: map[string]any{}, media: media}
	for k, v := range tokens {
		t.tokens[k] = v
	}
	return t, nil
}

// Prepare sets a function every conversation goes through before rendering.
func (t *FileTemplate) Prepare(f func([]Message) []Message) { t.prepare = f }

// Own is the engine's template, which renders when the file's is not wanted.
func (t *FileTemplate) Own() Template { return t.own }

func (t *FileTemplate) Render(msgs []Message, opt Options) (string, error) {
	if len(msgs) == 0 {
		return "", fmt.Errorf("chat: the template reads the first message of an empty conversation")
	}
	if t.prepare != nil {
		msgs = t.prepare(msgs)
	}
	messages := make([]any, len(msgs))
	for i, m := range msgs {
		messages[i] = messageValue(m)
	}
	var tools any // None when there are none, as transformers passes it
	if len(opt.Tools) > 0 {
		list := make([]any, len(opt.Tools))
		for i, tool := range opt.Tools {
			v, err := toolValue(tool)
			if err != nil {
				return "", err
			}
			list[i] = v
		}
		tools = list
	}
	vars := map[string]any{
		"messages":              messages,
		"tools":                 tools,
		"add_generation_prompt": opt.AddGenerationPrompt,
		"enable_thinking":       opt.EnableThinking,
	}
	for k, v := range t.tokens {
		vars[k] = v
	}
	out, err := t.tpl.Render(vars)
	if err != nil {
		return "", fmt.Errorf("chat: the model's template: %w", err)
	}
	if t.media != nil {
		out = t.media.Replace(out)
	}
	return out, nil
}

func (t *FileTemplate) ParseCalls(text string) (string, []ToolCall, error) {
	return t.own.ParseCalls(text)
}

func (t *FileTemplate) CallOpen() string { return t.own.CallOpen() }

func (t *FileTemplate) ReasoningMarkers() (string, string) { return t.own.ReasoningMarkers() }

// messageValue is one message the way transformers hands it to a template. A
// turn carrying pictures or recordings has its content as a list of parts —
// the pictures, then the recordings, then the text, which is the order the
// engines' own templates write them in — and a plain string otherwise.
func messageValue(m Message) *jinja.Dict {
	d := jinja.NewDict()
	d.Set("role", m.Role)
	if len(m.Images)+len(m.Audio) > 0 {
		var parts []any
		for range m.Images {
			parts = append(parts, part("image"))
		}
		for range m.Audio {
			parts = append(parts, part("audio"))
		}
		if m.Content != "" {
			p := part("text")
			p.Set("text", m.Content)
			parts = append(parts, p)
		}
		d.Set("content", parts)
	} else {
		d.Set("content", m.Content)
	}
	if len(m.ToolCalls) > 0 {
		calls := make([]any, len(m.ToolCalls))
		for i, c := range m.ToolCalls {
			call := jinja.NewDict()
			if c.ID != "" {
				call.Set("id", c.ID)
			}
			call.Set("type", "function")
			fn := jinja.NewDict()
			fn.Set("name", c.Name)
			args := c.Arguments
			if args == nil {
				args = map[string]any{}
			}
			fn.Set("arguments", jinja.FromGo(args))
			call.Set("function", fn)
			calls[i] = call
		}
		d.Set("tool_calls", calls)
	}
	if m.Name != "" {
		d.Set("name", m.Name)
	}
	if m.ToolCallID != "" {
		d.Set("tool_call_id", m.ToolCallID)
	}
	return d
}

func part(kind string) *jinja.Dict {
	p := jinja.NewDict()
	p.Set("type", kind)
	return p
}

// toolValue is one declaration as OpenAI spells it. A function taking no
// parameters has no "parameters" key, rather than a null one: templates test
// for the key.
func toolValue(t Tool) (*jinja.Dict, error) {
	fn := jinja.NewDict()
	fn.Set("name", t.Name)
	fn.Set("description", t.Description)
	for _, s := range []struct {
		key    string
		schema *Schema
	}{{"parameters", t.Parameters}, {"response", t.Response}} {
		if s.schema == nil {
			continue
		}
		raw, err := json.Marshal(s.schema)
		if err != nil {
			return nil, err
		}
		v, err := jinja.ParseJSON(raw)
		if err != nil {
			return nil, err
		}
		fn.Set(s.key, v)
	}
	d := jinja.NewDict()
	d.Set("type", "function")
	d.Set("function", fn)
	return d, nil
}
