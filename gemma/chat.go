package gemma

// The chat template, without Jinja.
//
// The GGUF carries eighteen kilobytes of Jinja under tokenizer.chat_template.
// Most of it serves tool declarations, tool calls and content-part arrays; a
// text conversation walks a path that fits on one screen, and that path is what
// is written out here. The fixture in testdata/gemma/chat holds what Jinja
// itself makes of that template, so this file is checked against the original
// rather than against a reading of it.
//
// What is not covered — images, audio, video, and the reasoning channel of a
// past answer — is refused rather than approximated.

import (
	"fmt"
	"strings"
)

// Message is one turn of the conversation. Role is "system", "developer",
// "user", "assistant", "model" or "tool"; the template renames "assistant" to
// "model" and treats "developer" as "system".
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Name is the function a tool result came from.
	Name string `json:"name,omitempty"`
	// ToolCalls are the calls a model turn made.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID ties a tool result to the call it answers.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type ChatOptions struct {
	// EnableThinking opens a system turn carrying <|think|>, even when the
	// conversation has no system message.
	EnableThinking bool
	// EmptyThought closes the generation prompt with a thought channel that is
	// opened and closed at once. It is the one rule the two checkpoints spell
	// differently: the 12B's template writes it whenever thinking is off, E2B's
	// template has no such line, and Config.EmptyThought says which of the two
	// the file being run is.
	EmptyThought bool
	// Tools are the functions declared to the model. They are written into the
	// system turn, which the template opens for them even when the
	// conversation has no system message.
	Tools []Tool
	// AddGenerationPrompt appends the empty model turn the model is meant to
	// continue. A caller rendering a conversation for training rather than for
	// generation leaves it false.
	AddGenerationPrompt bool
}

// The markers, spelled once.
const (
	bosPiece     = "<bos>"
	turnOpen     = "<|turn>"
	turnClose    = "<turn|>\n"
	thinkPiece   = "<|think|>\n"
	channelOpen  = "<|channel>"
	channelClose = "<channel|>"
	// emptyThought is what the 12B's template appends to a generation prompt
	// when thinking is off.
	emptyThought = channelOpen + "thought\n" + channelClose
	// emptyThoughtJinja is the same thing as the template spells it, where the
	// newline is still two characters. It is what LoadConfig looks for.
	emptyThoughtJinja = channelOpen + `thought\n` + channelClose
	roleSystem        = "system"
	roleDeveloper     = "developer"
	roleUser          = "user"
	roleAssistant     = "assistant"
	roleModel         = "model"
	roleTool          = "tool"
	toolOpen          = "<|tool>"
	toolClose         = "<tool|>"
	toolCallOpen      = "<|tool_call>"
	toolCallClose     = "<tool_call|>"
	toolResponseOpen  = "<|tool_response>"
	toolResponseClose = "<tool_response|>"
	quote             = `<|"|>`
)

// RenderChat writes the conversation the way Gemma's template does, leading
// <bos> included. The caller therefore encodes the result with addBOS false and
// parseSpecial true.
func RenderChat(messages []Message, opt ChatOptions) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("gemma: the template reads the first message of an empty conversation")
	}
	for i, m := range messages {
		switch m.Role {
		case roleUser, roleAssistant, roleModel:
		case roleTool:
			// A result answers a call, so a call has to come before it.
			if i == 0 {
				return "", fmt.Errorf("gemma: message 0 is a tool result, and a result answers a call that came before it")
			}
			if previous := messages[i-1]; previous.Role != roleTool && len(previous.ToolCalls) == 0 {
				return "", fmt.Errorf("gemma: message %d is a tool result, but message %d made no call", i, i-1)
			}
		case roleSystem, roleDeveloper:
			if i != 0 {
				return "", fmt.Errorf("gemma: message %d is a %s message, and the template opens its system turn from the first message only", i, m.Role)
			}
		default:
			return "", fmt.Errorf("gemma: message %d has role %q, which this renderer does not implement", i, m.Role)
		}
	}

	var b strings.Builder
	b.WriteString(bosPiece)

	rest := messages
	leadingSystem := messages[0].Role == roleSystem || messages[0].Role == roleDeveloper
	if opt.EnableThinking || leadingSystem || len(opt.Tools) > 0 {
		b.WriteString(turnOpen + roleSystem + "\n")
		if opt.EnableThinking {
			b.WriteString(thinkPiece)
		}
		if leadingSystem {
			b.WriteString(strings.TrimSpace(messages[0].Content))
			rest = messages[1:]
		}
		for _, tool := range opt.Tools {
			if err := writeDeclaration(&b, tool); err != nil {
				return "", err
			}
		}
		b.WriteString(turnClose)
	}

	previous := ""
	// lastWritten records what the loop wrote last, because the way a turn is
	// closed — and whether a generation prompt opens a new one — depends on it.
	lastWritten := ""
	for i, m := range rest {
		if m.Role == roleTool {
			continue // written by the call it answers
		}
		role := m.Role
		if role == roleAssistant {
			role = roleModel
		}
		// Two answers in a row are one model turn: the second opens no header.
		if !(role == roleModel && previous == roleAssistant) {
			b.WriteString(turnOpen + role + "\n")
		}
		for _, call := range m.ToolCalls {
			writeCall(&b, call)
			lastWritten = "tool_call"
		}
		// The results follow the message that called for them, inside its turn.
		answered := false
		for _, follow := range rest[i+1:] {
			if follow.Role != roleTool {
				break
			}
			name := follow.Name
			if name == "" {
				name = nameOfCall(m.ToolCalls, follow.ToolCallID)
			}
			writeResponse(&b, name, follow.Content)
			answered = true
			lastWritten = "tool_response"
		}
		content := m.Content
		if role == roleModel {
			content = StripThinking(content)
		} else {
			content = strings.TrimSpace(content)
		}
		b.WriteString(content)
		if content != "" {
			lastWritten = "content"
		}
		switch {
		case lastWritten == "tool_call" && !answered:
			// A call with no result yet: the template leaves the response open.
			b.WriteString(toolResponseOpen)
		case answered && content == "":
			// The turn stays open: the model speaks on after its results.
		default:
			b.WriteString(turnClose)
			lastWritten = "turn"
		}
		previous = m.Role
	}

	if opt.AddGenerationPrompt && lastWritten != "tool_call" && lastWritten != "tool_response" {
		b.WriteString(turnOpen + roleModel + "\n")
		if opt.EmptyThought && !opt.EnableThinking {
			b.WriteString(emptyThought)
		}
	}
	return b.String(), nil
}

// StripThinking removes the thinking channel from an answer that is being fed
// back as context. The template cuts the text on <channel|> and keeps, from
// each piece that holds a <|channel> marker, only what came before it — so an
// unclosed channel swallows everything after it, and the result is trimmed.
func StripThinking(text string) string {
	var b strings.Builder
	for _, part := range strings.Split(text, channelClose) {
		if i := strings.Index(part, channelOpen); i >= 0 {
			b.WriteString(part[:i])
			continue
		}
		b.WriteString(part)
	}
	return strings.TrimSpace(b.String())
}
