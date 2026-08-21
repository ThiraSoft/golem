package qwen

// The chat template, without Jinja.
//
// The GGUF carries the template under tokenizer.chat_template. It is not long,
// but three of its rules are not the ones a reading of the output would
// suggest, and all three are reproduced here: nothing is trimmed, the empty
// thought block goes to an assistant turn chosen by a backwards scan rather
// than to the last one, and consecutive tool results share a single user turn.
//
// The fixture in testdata/qwen/chat holds what Jinja itself makes of that
// template, so this file is checked against the original rather than against a
// reading of it. ref/qwen/dump_chats.py says where that fixture and llama.cpp
// part company, and why this follows the fixture.

import (
	"fmt"
	"strings"

	"github.com/ThiraSoft/golem/chat"
)

// The markers, spelled once.
const (
	imStart       = "<|im_start|>"
	imEnd         = "<|im_end|>\n"
	toolCallOpen  = "<tool_call>"
	toolCallClose = "</tool_call>"
	responseOpen  = "<tool_response>"
	responseClose = "</tool_response>"
	thinkOpen     = "<think>"
	thinkClose    = "</think>"
	// emptyThought is what the generation prompt carries when thinking is off.
	emptyThought  = thinkOpen + "\n\n" + thinkClose + "\n\n"
	roleSystem    = "system"
	roleDeveloper = "developer"
	roleUser      = "user"
	roleAssistant = "assistant"
	roleTool      = "tool"

	toolsPreamble = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
	toolsClosing  = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
)

// RenderChat writes the conversation the way Qwen3's template does. There is no
// leading BOS — the checkpoint has none — but the turn markers are special
// tokens, so the caller still encodes with addBOS false and parseSpecial true.
func RenderChat(msgs []chat.Message, opt chat.Options) (string, error) {
	if len(msgs) == 0 {
		return "", fmt.Errorf("qwen: the template reads the first message of an empty conversation")
	}
	// The template has no developer role and would drop such a message in
	// silence. It stands for the system message, and is read as one.
	msgs = append([]chat.Message(nil), msgs...)
	for i := range msgs {
		if msgs[i].Role == roleDeveloper {
			msgs[i].Role = roleSystem
		}
	}
	for i, m := range msgs {
		switch m.Role {
		case roleUser, roleAssistant, roleSystem:
		case roleTool:
			// A result answers a call, so a call has to come before it.
			if i == 0 {
				return "", fmt.Errorf("qwen: message 0 is a tool result, and a result answers a call that came before it")
			}
			if previous := msgs[i-1]; previous.Role != roleTool && len(previous.ToolCalls) == 0 {
				return "", fmt.Errorf("qwen: message %d is a tool result, but message %d made no call", i, i-1)
			}
		default:
			return "", fmt.Errorf("qwen: message %d has role %q, which this renderer does not implement", i, m.Role)
		}
	}

	var b strings.Builder
	leadingSystem := msgs[0].Role == roleSystem
	if len(opt.Tools) > 0 {
		// The tools turn is a system turn, opened for them whether or not the
		// conversation has a system message of its own.
		b.WriteString(imStart + roleSystem + "\n")
		if leadingSystem {
			b.WriteString(msgs[0].Content + "\n\n")
		}
		b.WriteString(toolsPreamble)
		for _, tool := range opt.Tools {
			b.WriteString("\n")
			writeDeclaration(&b, tool)
		}
		b.WriteString(toolsClosing + imEnd)
	} else if leadingSystem {
		b.WriteString(imStart + roleSystem + "\n" + msgs[0].Content + imEnd)
	}

	lastQuery := lastQueryIndex(msgs)
	for i, m := range msgs {
		switch m.Role {
		case roleSystem:
			// The first one opened the turn above; a later one is a plain turn.
			if i != 0 {
				b.WriteString(imStart + roleSystem + "\n" + m.Content + imEnd)
			}
		case roleUser:
			b.WriteString(imStart + roleUser + "\n" + m.Content + imEnd)
		case roleAssistant:
			writeAssistant(&b, m, i, i == len(msgs)-1, lastQuery)
		case roleTool:
			// Consecutive results share one user turn: the first opens it and
			// the last closes it.
			if i == 0 || msgs[i-1].Role != roleTool {
				b.WriteString(imStart + roleUser)
			}
			b.WriteString("\n" + responseOpen + "\n" + m.Content + "\n" + responseClose)
			if i == len(msgs)-1 || msgs[i+1].Role != roleTool {
				b.WriteString(imEnd)
			}
		}
	}

	if opt.AddGenerationPrompt {
		b.WriteString(imStart + roleAssistant + "\n")
		if !opt.EnableThinking {
			b.WriteString(emptyThought)
		}
	}
	return b.String(), nil
}

// lastQueryIndex is the index of the last user message that is not itself a
// tool result fed back as one. The template walks the conversation backwards to
// find it, and it is what decides which assistant turns keep their reasoning.
func lastQueryIndex(msgs []chat.Message) int {
	last := len(msgs) - 1
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		wrapped := strings.HasPrefix(m.Content, responseOpen) &&
			strings.HasSuffix(m.Content, responseClose)
		if m.Role == roleUser && !wrapped {
			return i
		}
	}
	return last
}

// writeAssistant writes one model turn. An answer that came after the last real
// question keeps its reasoning — or, if it has none and is the final turn, gets
// an empty block; an answer from earlier in the conversation loses it, which is
// what keeps a long conversation from carrying every thought the model ever had.
func writeAssistant(b *strings.Builder, m chat.Message, index int, last bool, lastQuery int) {
	content, reasoning := splitThinking(m.Content)
	b.WriteString(imStart + roleAssistant + "\n")
	if index > lastQuery && (last || strings.TrimSpace(reasoning) != "") {
		b.WriteString(thinkOpen + "\n" + strings.Trim(reasoning, "\n") + "\n" + thinkClose + "\n\n")
		content = strings.TrimLeft(content, "\n")
	}
	b.WriteString(content)
	for i, call := range m.ToolCalls {
		if (i == 0 && content != "") || i > 0 {
			b.WriteString("\n")
		}
		writeCall(b, call)
	}
	b.WriteString(imEnd)
}

// splitThinking separates an answer's reasoning from what it said. The template
// cuts on the last </think> and, in what came before, on the last <think>.
func splitThinking(text string) (content, reasoning string) {
	at := strings.LastIndex(text, thinkClose)
	if at < 0 {
		return text, ""
	}
	content = strings.TrimLeft(text[at+len(thinkClose):], "\n")
	reasoning = strings.TrimRight(text[:at], "\n")
	if open := strings.LastIndex(reasoning, thinkOpen); open >= 0 {
		reasoning = reasoning[open+len(thinkOpen):]
	}
	return content, strings.TrimLeft(reasoning, "\n")
}

// StripThinking removes the reasoning from an answer being fed back as context.
func StripThinking(text string) string {
	content, _ := splitThinking(text)
	return content
}
