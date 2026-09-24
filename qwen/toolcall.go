package qwen

// Reading back what the model wrote.
//
// Qwen3's calls are JSON inside XML tags, and Qwen3.5's are tags all the way
// down, a function and its parameters. Both make this a good deal shorter
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
	if trimmed := strings.TrimSpace(body); strings.HasPrefix(trimmed, functionOpen) {
		return readXMLCall(trimmed)
	}
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

// The markers of the other spelling, the one Qwen3.5's template teaches.
const (
	functionOpen   = "<function="
	functionClose  = "</function>"
	parameterOpen  = "<parameter="
	parameterClose = "</parameter>"
)

// readXMLCall reads a call written the way Qwen3.5's template writes one:
//
//	<function=weather>
//	<parameter=city>
//	Lyon
//	</parameter>
//	</function>
//
// The template writes a string as it is and anything else through tojson, and
// nothing here says which a parameter was declared as. So a value that reads as
// JSON is taken as JSON and any other is a string, which is what the template
// would have written for either.
func readXMLCall(body string) (chat.ToolCall, error) {
	rest := body[len(functionOpen):]
	gt := strings.IndexByte(rest, '>')
	if gt < 0 {
		return chat.ToolCall{}, fmt.Errorf("qwen: a function tag that never closes")
	}
	name := strings.TrimSpace(rest[:gt])
	if name == "" {
		return chat.ToolCall{}, fmt.Errorf("qwen: a tool call with no function name")
	}
	rest = rest[gt+1:]
	end := strings.LastIndex(rest, functionClose)
	if end < 0 {
		return chat.ToolCall{}, fmt.Errorf("qwen: a function block that never closes")
	}
	if tail := strings.TrimSpace(rest[end+len(functionClose):]); tail != "" {
		return chat.ToolCall{}, fmt.Errorf("qwen: %q follows a function block", tail)
	}
	rest = rest[:end]

	args := map[string]any{}
	for {
		rest = strings.TrimLeft(rest, " \t\r\n")
		if rest == "" {
			return chat.ToolCall{Name: name, Arguments: args}, nil
		}
		if !strings.HasPrefix(rest, parameterOpen) {
			return chat.ToolCall{}, fmt.Errorf("qwen: %q inside a function block is not a parameter", rest)
		}
		rest = rest[len(parameterOpen):]
		gt := strings.IndexByte(rest, '>')
		if gt < 0 {
			return chat.ToolCall{}, fmt.Errorf("qwen: a parameter tag that never closes")
		}
		key := strings.TrimSpace(rest[:gt])
		if key == "" {
			return chat.ToolCall{}, fmt.Errorf("qwen: a parameter with no name")
		}
		rest = rest[gt+1:]
		stop := strings.Index(rest, parameterClose)
		if stop < 0 {
			return chat.ToolCall{}, fmt.Errorf("qwen: parameter %q never closes", key)
		}
		args[key] = parameterValue(rest[:stop])
		rest = rest[stop+len(parameterClose):]
	}
}

// parameterValue undoes the newline the template puts on each side of a value,
// and reads it as JSON when it is some.
func parameterValue(raw string) any {
	raw = strings.TrimPrefix(raw, "\n")
	raw = strings.TrimSuffix(raw, "\n")
	var v any
	if err := json.Unmarshal([]byte(raw), &v); err == nil {
		if _, isString := v.(string); !isString {
			return v
		}
	}
	return raw
}
