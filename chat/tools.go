package chat

// The tool vocabulary of the chat template, and the JSON the OpenAI API
// spells it with.
//
// One conversion is worth stating. On the wire a call's arguments are a
// *string* holding JSON, because that is what OpenAI's API settled on; the
// template walks a mapping. So a call holds the decoded mapping and converts
// at both edges: it reads a string or an object, and it always writes a
// string, which is what a client feeding the call back expects.

import (
	"encoding/json"
	"fmt"
)

// Tool is one function the model may call.
type Tool struct {
	Name        string
	Description string
	Parameters  *Schema // nil when the function takes none
	Response    *Schema // nil unless the caller declares one
}

// Schema is the subset of JSON Schema the template reads.
type Schema struct {
	Type        string             `json:"type,omitempty"`
	Description string             `json:"description,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	Enum        []string           `json:"enum,omitempty"`
	Nullable    bool               `json:"nullable,omitempty"`
}

type toolWire struct {
	Type     string `json:"type"`
	Function struct {
		Name        string  `json:"name"`
		Description string  `json:"description"`
		Parameters  *Schema `json:"parameters"`
		Response    *Schema `json:"response"`
	} `json:"function"`
}

func (t *Tool) UnmarshalJSON(raw []byte) error {
	var w toolWire
	if err := json.Unmarshal(raw, &w); err != nil {
		return err
	}
	if w.Type != "" && w.Type != "function" {
		return fmt.Errorf("chat: tool type %q is not implemented; only functions are", w.Type)
	}
	if w.Function.Name == "" {
		return fmt.Errorf("chat: a tool with no function name")
	}
	*t = Tool{
		Name:        w.Function.Name,
		Description: w.Function.Description,
		Parameters:  w.Function.Parameters,
		Response:    w.Function.Response,
	}
	return nil
}

func (t Tool) MarshalJSON() ([]byte, error) {
	var w toolWire
	w.Type = "function"
	w.Function.Name = t.Name
	w.Function.Description = t.Description
	w.Function.Parameters = t.Parameters
	w.Function.Response = t.Response
	return json.Marshal(w)
}

// ToolCall is one call the model made, or one being fed back as context.
type ToolCall struct {
	ID        string // the client's identifier; the template never renders it
	Name      string
	Arguments map[string]any
	// Index numbers a call inside one streamed delta, and is written only
	// when HasIndex says the caller means it: zero is a real index.
	Index    int
	HasIndex bool
}

type callWire struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Index    *int   `json:"index,omitempty"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

func (c *ToolCall) UnmarshalJSON(raw []byte) error {
	var w callWire
	if err := json.Unmarshal(raw, &w); err != nil {
		return err
	}
	if w.Type != "" && w.Type != "function" {
		return fmt.Errorf("chat: tool call type %q is not implemented", w.Type)
	}
	args, err := decodeArguments(w.Function.Arguments)
	if err != nil {
		return err
	}
	*c = ToolCall{ID: w.ID, Name: w.Function.Name, Arguments: args}
	if w.Index != nil {
		c.Index, c.HasIndex = *w.Index, true
	}
	return nil
}

// decodeArguments reads the two spellings the wire allows: a JSON object, and
// a string holding one.
func decodeArguments(raw json.RawMessage) (map[string]any, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, err
		}
		if text == "" {
			return nil, nil
		}
		raw = json.RawMessage(text)
	}
	var args map[string]any
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("chat: tool call arguments are not a JSON object: %w", err)
	}
	return args, nil
}

func (c ToolCall) MarshalJSON() ([]byte, error) {
	args := c.Arguments
	if args == nil {
		args = map[string]any{}
	}
	encoded, err := json.Marshal(args)
	if err != nil {
		return nil, err
	}
	var w callWire
	w.ID = c.ID
	w.Type = "function"
	if c.HasIndex {
		index := c.Index
		w.Index = &index
	}
	w.Function.Name = c.Name
	w.Function.Arguments, err = json.Marshal(string(encoded))
	if err != nil {
		return nil, err
	}
	return json.Marshal(w)
}
