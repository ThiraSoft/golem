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
// What is not covered — tools, tool calls, tool responses, images, audio,
// video, and the reasoning channel of a past answer — is refused rather than
// approximated.

import (
	"fmt"
	"strings"
)

// Message is one turn of the conversation. Role is "system", "developer",
// "user", "assistant" or "model"; the template renames "assistant" to "model"
// and treats "developer" as "system".
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatOptions struct {
	// EnableThinking opens a system turn carrying <|think|>, even when the
	// conversation has no system message.
	EnableThinking bool
	// AddGenerationPrompt appends the empty model turn the model is meant to
	// continue. A caller rendering a conversation for training rather than for
	// generation leaves it false.
	AddGenerationPrompt bool
}

// The markers, spelled once.
const (
	bosPiece      = "<bos>"
	turnOpen      = "<|turn>"
	turnClose     = "<turn|>\n"
	thinkPiece    = "<|think|>\n"
	channelOpen   = "<|channel>"
	channelClose  = "<channel|>"
	roleSystem    = "system"
	roleDeveloper = "developer"
	roleUser      = "user"
	roleAssistant = "assistant"
	roleModel     = "model"
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
	if opt.EnableThinking || leadingSystem {
		b.WriteString(turnOpen + roleSystem + "\n")
		if opt.EnableThinking {
			b.WriteString(thinkPiece)
		}
		if leadingSystem {
			b.WriteString(strings.TrimSpace(messages[0].Content))
			rest = messages[1:]
		}
		b.WriteString(turnClose)
	}

	previous := ""
	for _, m := range rest {
		role := m.Role
		if role == roleAssistant {
			role = roleModel
		}
		// Two answers in a row are one model turn: the second opens no header.
		if !(role == roleModel && previous == roleAssistant) {
			b.WriteString(turnOpen + role + "\n")
		}
		if role == roleModel {
			b.WriteString(StripThinking(m.Content))
		} else {
			b.WriteString(strings.TrimSpace(m.Content))
		}
		b.WriteString(turnClose)
		previous = m.Role
	}

	if opt.AddGenerationPrompt {
		b.WriteString(turnOpen + roleModel + "\n")
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
