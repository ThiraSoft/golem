package qwen

// Reading back what the model wrote.
//
// Qwen's calls are JSON inside XML tags, which makes this a good deal shorter
// than Gemma's scanner. What it keeps from Gemma is the rule that matters: it
// refuses what it cannot read rather than returning half a call, because a
// half-read call would be executed.

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ThiraSoft/golem/chat"
)

// ParseToolCalls splits an answer into the prose before the first call and the
// calls that followed.
func ParseToolCalls(text string) (string, []chat.ToolCall, error) {
	head := strings.Index(text, toolCallOpen)
	if head < 0 {
		return text, nil, nil
	}
	before := strings.TrimSpace(text[:head])
	rest := text[head:]

	var calls []chat.ToolCall
	for {
		rest = strings.TrimLeft(rest, " \t\r\n")
		if rest == "" {
			return before, calls, nil
		}
		if !strings.HasPrefix(rest, toolCallOpen) {
			// The template writes nothing after the last call, so anything
			// here means the answer was not read the way it was written.
			return before, nil, fmt.Errorf("qwen: %q follows a tool call and is neither a call nor the end of the answer", rest)
		}
		rest = rest[len(toolCallOpen):]
		end := strings.Index(rest, toolCallClose)
		if end < 0 {
			return before, nil, fmt.Errorf("qwen: a tool call that never closes")
		}
		body := rest[:end]
		rest = rest[end+len(toolCallClose):]

		call, err := readCall(body)
		if err != nil {
			return before, nil, err
		}
		calls = append(calls, call)
	}
}

// readCall reads the object between the tags. The template always writes both
// keys — the name as a string and the arguments as a mapping — so a call
// missing either is a call this cannot vouch for.
func readCall(body string) (chat.ToolCall, error) {
	var wire struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(body), &wire); err != nil {
		return chat.ToolCall{}, fmt.Errorf("qwen: a tool call that is not JSON: %w", err)
	}
	if wire.Name == "" {
		return chat.ToolCall{}, fmt.Errorf("qwen: a tool call with no function name")
	}
	if len(wire.Arguments) == 0 {
		return chat.ToolCall{}, fmt.Errorf("qwen: a tool call with no arguments block")
	}
	args := map[string]any{}
	if string(wire.Arguments) != "null" {
		if err := json.Unmarshal(wire.Arguments, &args); err != nil {
			return chat.ToolCall{}, fmt.Errorf("qwen: tool call arguments are not a JSON object: %w", err)
		}
	}
	return chat.ToolCall{Name: wire.Name, Arguments: args}, nil
}
