# OpenAI-Compatible Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a mapped Gemma GGUF over an OpenAI-compatible HTTP API that declares tools to the model and reports the calls it makes.

**Architecture:** `gemma/` grows the tool path of the chat template — declarations, calls and responses rendered exactly as the GGUF's Jinja renders them, pinned to a Jinja-produced fixture — plus a scanner that reads a call back out of the model's output. `cmd/serve` puts an HTTP layer over that: one mapped model behind a mutex, and a KV cache reused across stateless requests by matching the common token prefix.

**Tech Stack:** Go standard library only. No cgo, no dependencies. Python with Jinja2 is used once, by hand, to record fixtures.

**Spec:** `docs/superpowers/specs/2026-08-20-openai-server-design.md`

## Global Constraints

- Code, comments, commit messages and documentation in English.
- Standard library only: no new entries in `go.mod`.
- No engine imports another engine; `cmd/serve` may import `gemma`, `token/bpe`, `sample`.
- Tests must not need weights on the machine, except tests explicitly guarded by `GOLEM_MODEL`.
- Fixtures are committed and never regenerated at test time.
- The two model files on this machine:
  - `GOLEM_MODEL=/mnt/data/LLMs_models/lmstudio-community/gemma-4-E2B-it-QAT-GGUF/gemma-4-E2B-it-QAT-Q4_0.gguf`
  - `GOLEM_MODEL_12B=/mnt/data/LLMs_models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf`
- Run all tests with `go test ./...`; never push.

---

### Task 1: The tool types and their JSON

**Files:**
- Create: `gemma/tools.go`
- Test: `gemma/tools_test.go`

**Interfaces:**
- Produces: `gemma.Tool`, `gemma.Schema`, `gemma.ToolCall`, with `UnmarshalJSON` on `Tool` and on `ToolCall` accepting the OpenAI wire shape, and `MarshalJSON` on `ToolCall` writing it back.

The OpenAI wire shape is `{"type":"function","function":{"name","description","parameters"}}` for a tool, and `{"id","type":"function","function":{"name","arguments"}}` for a call, where `arguments` is a **string** holding JSON. The template, however, walks an arguments *mapping*. So `ToolCall` holds a decoded `map[string]any` and converts at the edges: it accepts a string or an object when decoding, and always writes a string when encoding, which is what clients expect.

- [ ] **Step 1: Write the failing test**

```go
package gemma

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestToolDecodesTheOpenAIShape(t *testing.T) {
	var tool Tool
	raw := `{"type":"function","function":{"name":"weather",
		"description":"Today's weather.",
		"parameters":{"type":"object",
			"properties":{"city":{"type":"string","description":"Where."},
			              "unit":{"type":"string","enum":["c","f"]}},
			"required":["city"]}}}`
	if err := json.Unmarshal([]byte(raw), &tool); err != nil {
		t.Fatal(err)
	}
	if tool.Name != "weather" || tool.Description != "Today's weather." {
		t.Fatalf("%+v", tool)
	}
	if tool.Parameters == nil || tool.Parameters.Type != "object" {
		t.Fatalf("parameters %+v", tool.Parameters)
	}
	if got := tool.Parameters.Properties["city"].Description; got != "Where." {
		t.Fatalf("city description %q", got)
	}
	if got := tool.Parameters.Properties["unit"].Enum; !reflect.DeepEqual(got, []string{"c", "f"}) {
		t.Fatalf("enum %v", got)
	}
	if got := tool.Parameters.Required; !reflect.DeepEqual(got, []string{"city"}) {
		t.Fatalf("required %v", got)
	}
}

func TestToolRefusesWhatIsNotAFunction(t *testing.T) {
	var tool Tool
	err := json.Unmarshal([]byte(`{"type":"retrieval"}`), &tool)
	if err == nil {
		t.Fatal("a tool that is not a function should be refused rather than silently dropped")
	}
}

func TestToolCallArgumentsDecodeFromAStringOrAnObject(t *testing.T) {
	want := map[string]any{"city": "Lyon", "days": float64(3)}
	for _, raw := range []string{
		`{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Lyon\",\"days\":3}"}}`,
		`{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Lyon","days":3}}}`,
	} {
		var call ToolCall
		if err := json.Unmarshal([]byte(raw), &call); err != nil {
			t.Fatal(err)
		}
		if call.ID != "call_1" || call.Name != "weather" {
			t.Fatalf("%+v", call)
		}
		if !reflect.DeepEqual(call.Arguments, want) {
			t.Fatalf("arguments %#v", call.Arguments)
		}
	}
}

func TestToolCallMarshalsArgumentsAsAString(t *testing.T) {
	raw, err := json.Marshal(ToolCall{ID: "call_1", Name: "weather",
		Arguments: map[string]any{"city": "Lyon"}})
	if err != nil {
		t.Fatal(err)
	}
	var back struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if back.Type != "function" || back.Function.Arguments != `{"city":"Lyon"}` {
		t.Fatalf("%s", raw)
	}
}

// Arguments with no key are an empty object, not null: a client feeding the
// call back must get valid JSON.
func TestToolCallWithNoArgumentsMarshalsAsAnEmptyObject(t *testing.T) {
	raw, _ := json.Marshal(ToolCall{ID: "c", Name: "now"})
	if want := `"arguments":"{}"`; !contains(string(raw), want) {
		t.Fatalf("%s does not hold %s", raw, want)
	}
}

func contains(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (func() bool {
		for i := 0; i+len(needle) <= len(haystack); i++ {
			if haystack[i:i+len(needle)] == needle {
				return true
			}
		}
		return false
	})()
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./gemma/ -run 'Tool' -v`
Expected: FAIL — `undefined: Tool`.

- [ ] **Step 3: Write `gemma/tools.go`**

```go
package gemma

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
		return fmt.Errorf("gemma: tool type %q is not implemented; only functions are", w.Type)
	}
	if w.Function.Name == "" {
		return fmt.Errorf("gemma: a tool with no function name")
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
		return fmt.Errorf("gemma: tool call type %q is not implemented", w.Type)
	}
	args, err := decodeArguments(w.Function.Arguments)
	if err != nil {
		return err
	}
	*c = ToolCall{ID: w.ID, Name: w.Function.Name, Arguments: args}
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
		return nil, fmt.Errorf("gemma: tool call arguments are not a JSON object: %w", err)
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
	w.Function.Name = c.Name
	w.Function.Arguments, err = json.Marshal(string(encoded))
	if err != nil {
		return nil, err
	}
	return json.Marshal(w)
}
```

- [ ] **Step 4: Run the tests**

Run: `go test ./gemma/ -run 'Tool' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gemma/tools.go gemma/tools_test.go
git commit -m "The types a tool call is spelled with, on the wire and in the template"
```

---

### Task 2: Record what Jinja makes of the tool path

**Files:**
- Modify: `ref/gemma/dump_chats.py` (the `CASES` table, the render loop, the JSON written)
- Modify: `testdata/gemma/chat/cases.json`, `testdata/gemma/chat12/cases.json` (regenerated)
- Modify: `ref/gemma/README.md` (say the fixture now covers tools)

The fixture is the arbiter for Tasks 3 and 4. It is recorded first so those tasks have something to fail against.

**Interfaces:**
- Produces: fixture cases carrying a new `tools` field (OpenAI shape) alongside `messages`, and messages carrying `tool_calls`.

- [ ] **Step 1: Widen the case tuple**

`CASES` entries become five-tuples `(name, messages, thinking, add_generation_prompt, tools)`. Add `None` as the fifth element of every existing entry.

- [ ] **Step 2: Add the tool cases**

```python
WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Today's weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Which city."},
                "unit": {"type": "string", "description": "Scale.",
                         "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
NOW = {"type": "function",
       "function": {"name": "now", "description": "The current time."}}
TAGS = {
    "type": "function",
    "function": {
        "name": "tags",
        "description": "Tag a note.",
        "parameters": {
            "type": "object",
            "properties": {
                "labels": {"type": "array", "description": "The labels.",
                           "items": {"type": "string"}},
                "pinned": {"type": "boolean", "description": "Keep it on top."},
            },
        },
    },
}

TOOL_CASES = [
    ("tools_declared", [{"role": "user", "content": "weather in Lyon?"}],
     False, True, [WEATHER]),
    ("tools_no_parameters", [{"role": "user", "content": "what time is it?"}],
     False, True, [NOW]),
    ("tools_array_and_boolean", [{"role": "user", "content": "tag it"}],
     False, True, [TAGS]),
    ("tools_two_declarations", [{"role": "user", "content": "hi"}],
     False, True, [WEATHER, NOW]),
    ("tools_with_system", [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hi"},
    ], False, True, [WEATHER]),
    ("tools_thinking", [{"role": "user", "content": "hi"}], True, True, [WEATHER]),
    ("tool_call_unanswered", [
        {"role": "user", "content": "weather in Lyon?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "weather",
                          "arguments": {"city": "Lyon", "unit": "celsius"}}},
        ]},
    ], False, False, [WEATHER]),
    ("tool_call_answered", [
        {"role": "user", "content": "weather in Lyon?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "weather", "arguments": {"city": "Lyon"}}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "weather",
         "content": "18 degrees, clear"},
    ], False, True, [WEATHER]),
    ("tool_call_two_calls_answered", [
        {"role": "user", "content": "both please"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "weather", "arguments": {"city": "Lyon"}}},
            {"id": "call_2", "type": "function",
             "function": {"name": "now", "arguments": {}}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "weather",
         "content": "18 degrees"},
        {"role": "tool", "tool_call_id": "call_2", "name": "now",
         "content": "14:03"},
    ], False, True, [WEATHER, NOW]),
    ("tool_call_then_answer", [
        {"role": "user", "content": "weather in Lyon?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "weather", "arguments": {"city": "Lyon"}}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "weather",
         "content": "18 degrees"},
        {"role": "assistant", "content": "It is 18 degrees."},
        {"role": "user", "content": "and tomorrow?"},
    ], False, True, [WEATHER]),
    ("tool_call_argument_types", [
        {"role": "user", "content": "tag it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "tags",
                          "arguments": {"labels": ["a", "b"], "pinned": True,
                                        "count": 2}}},
        ]},
    ], False, False, [TAGS]),
]

CASES = CASES + TOOL_CASES
```

- [ ] **Step 3: Pass the tools through to Jinja and into the fixture**

In `main`, unpack five values and render with `tools=tools`; add `"tools": tools` to each recorded case.

```python
    for name, messages, thinking, add_generation_prompt, tools in CASES:
        try:
            text = template.render(
                messages=messages,
                bos_token="<bos>",
                enable_thinking=thinking,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
            )
        except TemplateError as e:
            raise SystemExit(f"{name}: {e}")
        cases.append({
            "name": name,
            "messages": messages,
            "tools": tools,
            "enable_thinking": thinking,
            "add_generation_prompt": add_generation_prompt,
            "rendered": text,
        })
```

- [ ] **Step 4: Record both fixtures**

```bash
python3 ref/gemma/dump_chats.py "$GOLEM_MODEL" testdata/gemma/chat/cases.json
python3 ref/gemma/dump_chats.py "$GOLEM_MODEL_12B" testdata/gemma/chat12/cases.json
```

Expected: `25 cases written to …` for each. Read three of the new `rendered`
strings by eye and check they hold `<|tool>declaration:`, `<|tool_call>call:`
and `<|tool_response>response:` — if a case rendered no tool markers at all,
the case is wrong, not the template.

- [ ] **Step 5: Run the existing tests**

Run: `go test ./gemma/ -run RenderChat`
Expected: FAIL on the new cases only — the old ones still pass, which shows the
regeneration changed nothing it should not have. Note which new cases fail;
Task 3 makes them pass.

- [ ] **Step 6: Commit**

```bash
git add ref/gemma/dump_chats.py ref/gemma/README.md testdata/gemma/chat/cases.json testdata/gemma/chat12/cases.json
git commit -m "Record what the template makes of tools, calls and responses"
```

---

### Task 3: Render tools, calls and responses

**Files:**
- Modify: `gemma/chat.go`
- Modify: `gemma/chat_test.go` (decode `tools` and `tool_calls` in `chatCase`)
- Create: `gemma/toolrender.go` (the declaration and argument formatting)

**Interfaces:**
- Consumes: `Tool`, `Schema`, `ToolCall` from Task 1; the fixture from Task 2.
- Produces: `Message.ToolCalls []ToolCall`, `Message.Name string`, the role `tool`, `ChatOptions.Tools []Tool`, and `RenderChat` rendering all of it.

The template's rules, restated so the implementer needs no Jinja:

- Keys of any mapping are walked sorted (Jinja's `dictsort`).
- `<|"|>` is the quote. A declaration's property keys are bare; a call's
  argument keys are bare; a response's keys are bare.
- Types are upper-cased inside quotes: `"string"` becomes `<|"|>STRING<|"|>`.
- One declaration: `declaration:NAME{description:<|"|>D<|"|>,parameters:{properties:{…},required:[<|"|>a<|"|>],type:<|"|>OBJECT<|"|>}}`, wrapped in `<|tool>` … `<tool|>`, trimmed.
- A property: `key{description:<|"|>D<|"|>,type:<|"|>STRING<|"|>}`, with `enum:[…]` before the type for a string, `items:{…}` for an array, `nullable:true`, and nested `properties`/`required` for an object. The comma appears between whatever parts are present, and `type` is always last.
- A call: `<|tool_call>call:NAME{k:v,…}<tool_call|>`, values formatted as
  strings quoted with `<|"|>`, booleans as `true`/`false`, numbers bare, arrays
  as `[…]`, objects as `{k:v}` with **bare** keys.
- A response: `<|tool_response>response:NAME{k:v}<tool_response|>`; a
  non-mapping result becomes `{value:…}`.
- Turn closing: after an assistant message whose calls were answered by
  following `tool` messages and whose own content is empty, the template writes
  **no** `<turn|>`; after an assistant message whose calls were *not* answered
  it writes a bare `<|tool_response>`; and `AddGenerationPrompt` adds no
  `<|turn>model` header when the last thing written was a call or a response.
- `tool` messages are consumed by the assistant message that called them and
  are never rendered on their own.

- [ ] **Step 1: Widen the test's case type**

```go
type chatCase struct {
	Name                string    `json:"name"`
	Messages            []Message `json:"messages"`
	Tools               []Tool    `json:"tools"`
	EnableThinking      bool      `json:"enable_thinking"`
	AddGenerationPrompt bool      `json:"add_generation_prompt"`
	Rendered            string    `json:"rendered"`
}
```

and pass `Tools: c.Tools` in `ChatOptions` inside `TestRenderChatMatchesTheTemplate`.

In `TestRenderChatRejectsWhatItCannotRender`, replace the assertion that a lone
`tool` message is refused — it now is refused for a different reason, that no
call precedes it:

```go
	if _, err := RenderChat([]Message{{Role: "tool", Name: "weather", Content: "x"}}, ChatOptions{}); err == nil {
		t.Fatal("a tool result with no call before it should be refused")
	}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./gemma/ -run RenderChat`
Expected: FAIL — `unknown field Tools`, then the tool cases mismatching.

- [ ] **Step 3: Widen `Message` and `ChatOptions`**

In `gemma/chat.go`:

```go
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
```

and add to `ChatOptions`:

```go
	// Tools are the functions declared to the model. They are written into the
	// system turn, which the template opens for them even with no system
	// message.
	Tools []Tool
```

with `roleTool = "tool"` beside the other role constants, and the markers:

```go
	toolOpen          = "<|tool>"
	toolClose         = "<tool|>"
	toolCallOpen      = "<|tool_call>"
	toolCallClose     = "<tool_call|>"
	toolResponseOpen  = "<|tool_response>"
	toolResponseClose = "<tool_response|>"
	quote             = `<|"|>`
```

- [ ] **Step 4: Write `gemma/toolrender.go`**

```go
package gemma

// Tools, as the template spells them.
//
// The syntax is the template's own and not JSON: <|"|> is the quote, keys are
// bare, types are upper-cased, and every mapping is walked in sorted key
// order because Jinja's dictsort does. Everything here exists to reproduce
// that, character for character.

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// quoted writes one string the way the template quotes it.
func quoted(text string) string { return quote + text + quote }

// upperQuoted writes a type name, which the template upper-cases.
func upperQuoted(text string) string { return quoted(strings.ToUpper(text)) }

// writeDeclaration writes one tool between <|tool> and <tool|>.
func writeDeclaration(b *strings.Builder, t Tool) error {
	if t.Name == "" {
		return fmt.Errorf("gemma: a tool with no name")
	}
	b.WriteString(toolOpen)
	b.WriteString("declaration:" + t.Name + "{description:" + quoted(t.Description))
	if t.Parameters != nil {
		b.WriteString(",parameters:{")
		writeSchemaBody(b, t.Parameters)
		b.WriteString("}")
	}
	if t.Response != nil {
		b.WriteString(",response:{")
		first := false
		if t.Response.Description != "" {
			b.WriteString("description:" + quoted(t.Response.Description))
			first = true
		}
		if t.Response.Type != "" {
			if first {
				b.WriteString(",")
			}
			b.WriteString("type:" + upperQuoted(t.Response.Type))
		}
		b.WriteString("}")
	}
	b.WriteString("}")
	b.WriteString(toolClose)
	return nil
}

// writeSchemaBody writes properties, required and type, in that order, which
// is the order the template writes them in.
func writeSchemaBody(b *strings.Builder, s *Schema) {
	first := false
	if len(s.Properties) > 0 {
		b.WriteString("properties:{")
		writeProperties(b, s.Properties, s.Required)
		b.WriteString("}")
		first = true
	}
	if len(s.Required) > 0 {
		if first {
			b.WriteString(",")
		}
		b.WriteString("required:[")
		for i, name := range s.Required {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(quoted(name))
		}
		b.WriteString("]")
		first = true
	}
	if s.Type != "" {
		if first {
			b.WriteString(",")
		}
		b.WriteString("type:" + upperQuoted(s.Type))
	}
}

// writeProperties writes one entry per property, sorted by name.
func writeProperties(b *strings.Builder, props map[string]*Schema, required []string) {
	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	sort.Strings(names)
	for i, name := range names {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(name + "{")
		writeProperty(b, props[name])
		b.WriteString("}")
	}
}

// writeProperty writes one property's body. type comes last, whatever else is
// there, and each part is separated from the one before it by a comma.
func writeProperty(b *strings.Builder, s *Schema) {
	comma := false
	sep := func() {
		if comma {
			b.WriteString(",")
		}
		comma = true
	}
	if s.Description != "" {
		sep()
		b.WriteString("description:" + quoted(s.Description))
	}
	kind := strings.ToLower(s.Type)
	switch {
	case kind == "string" && len(s.Enum) > 0:
		sep()
		b.WriteString("enum:[")
		for i, v := range s.Enum {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(quoted(v))
		}
		b.WriteString("]")
	case kind == "array" && s.Items != nil:
		sep()
		b.WriteString("items:{")
		writeItems(b, s.Items)
		b.WriteString("}")
	}
	if s.Nullable {
		sep()
		b.WriteString("nullable:true")
	}
	if kind == "object" {
		if len(s.Properties) > 0 {
			sep()
			b.WriteString("properties:{")
			writeProperties(b, s.Properties, s.Required)
			b.WriteString("}")
		}
		if len(s.Required) > 0 {
			sep()
			b.WriteString("required:[")
			for i, name := range s.Required {
				if i > 0 {
					b.WriteString(",")
				}
				b.WriteString(quoted(name))
			}
			b.WriteString("]")
		}
	}
	sep()
	b.WriteString("type:" + upperQuoted(s.Type))
}

// writeItems writes an array's item schema: the template walks its keys
// sorted, skipping the ones it has no value for.
func writeItems(b *strings.Builder, s *Schema) {
	parts := map[string]string{}
	if s.Description != "" {
		parts["description"] = quoted(s.Description)
	}
	if s.Type != "" {
		parts["type"] = upperQuoted(s.Type)
	}
	keys := make([]string, 0, len(parts))
	for k := range parts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for i, k := range keys {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(k + ":" + parts[k])
	}
}

// writeCall writes one call the model made.
func writeCall(b *strings.Builder, c ToolCall) {
	b.WriteString(toolCallOpen + "call:" + c.Name + "{")
	writeArgumentMap(b, c.Arguments)
	b.WriteString("}" + toolCallClose)
}

// writeResponse writes one tool result. A result that is not a mapping is
// wrapped, as the template wraps it.
func writeResponse(b *strings.Builder, name, content string) {
	b.WriteString(toolResponseOpen + "response:" + name + "{")
	b.WriteString("value:" + quoted(content))
	b.WriteString("}" + toolResponseClose)
}

// writeArgumentMap writes a mapping's entries with bare keys, sorted.
func writeArgumentMap(b *strings.Builder, args map[string]any) {
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for i, k := range keys {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(k + ":")
		writeArgument(b, args[k])
	}
}

// writeArgument writes one value: a string quoted, a boolean as a word, a
// number bare, a list between brackets, a mapping between braces.
func writeArgument(b *strings.Builder, v any) {
	switch v := v.(type) {
	case nil:
		b.WriteString("None")
	case string:
		b.WriteString(quoted(v))
	case bool:
		if v {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case float64:
		b.WriteString(formatNumber(v))
	case int:
		b.WriteString(strconv.Itoa(v))
	case []any:
		b.WriteString("[")
		for i, item := range v {
			if i > 0 {
				b.WriteString(",")
			}
			writeArgument(b, item)
		}
		b.WriteString("]")
	case map[string]any:
		b.WriteString("{")
		writeArgumentMap(b, v)
		b.WriteString("}")
	default:
		fmt.Fprintf(b, "%v", v)
	}
}

// formatNumber writes a number the way Python's str would: a whole float has
// no fractional part, because JSON gave us a float64 for what was written 3.
func formatNumber(f float64) string {
	if f == float64(int64(f)) {
		return strconv.FormatInt(int64(f), 10)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}
```

- [ ] **Step 5: Render them from `RenderChat`**

Replace the validation loop and the message loop in `gemma/chat.go`.

Validation: a `tool` message is legal only when the message before it is an
assistant message carrying calls, or another `tool` message.

```go
	for i, m := range messages {
		switch m.Role {
		case roleUser, roleAssistant, roleModel:
		case roleTool:
			if i == 0 {
				return "", fmt.Errorf("gemma: message 0 is a tool result, and a result answers a call that came before it")
			}
			previous := messages[i-1]
			if previous.Role != roleTool && len(previous.ToolCalls) == 0 {
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
```

The system turn opens when there are tools, even with no system message and no
thinking:

```go
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
```

The message loop skips `tool` messages, writes the calls of the message that
owns them, and closes the turn by the template's rule. `lastWritten` records
what the loop wrote last, because the generation prompt reads it.

```go
	previous := ""
	lastWritten := ""
	for i, m := range rest {
		if m.Role == roleTool {
			continue // consumed by the call it answers
		}
		role := m.Role
		if role == roleAssistant {
			role = roleModel
		}
		if !(role == roleModel && previous == roleAssistant) {
			b.WriteString(turnOpen + role + "\n")
		}
		for _, call := range m.ToolCalls {
			writeCall(&b, call)
			lastWritten = "tool_call"
		}
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
			b.WriteString(toolResponseOpen)
		case answered && content == "":
			// The turn stays open: the model speaks on after its results.
		default:
			b.WriteString(turnClose)
			lastWritten = "turn"
		}
		previous = m.Role
	}

	if opt.AddGenerationPrompt {
		if lastWritten != "tool_call" && lastWritten != "tool_response" {
			b.WriteString(turnOpen + roleModel + "\n")
			if opt.EmptyThought && !opt.EnableThinking {
				b.WriteString(emptyThought)
			}
		}
	}
```

with

```go
// nameOfCall finds which function a result answers, when the result names only
// the call's identifier.
func nameOfCall(calls []ToolCall, id string) string {
	for _, c := range calls {
		if c.ID == id {
			return c.Name
		}
	}
	return "unknown"
}
```

- [ ] **Step 6: Run the tests and close the gap**

Run: `go test ./gemma/ -run RenderChat -v`
Expected: PASS on all 25 cases of both fixtures. Any mismatch is printed as two
quoted strings; the difference is a character of the template, so read the
Jinja rather than adjusting the fixture. The fixture is never edited by hand.

- [ ] **Step 7: Commit**

```bash
git add gemma/chat.go gemma/toolrender.go gemma/chat_test.go
git commit -m "The template renders tools, the calls they draw, and their results"
```

---

### Task 4: Read a call back out of the model's output

**Files:**
- Create: `gemma/toolcall.go`
- Test: `gemma/toolcall_test.go`

**Interfaces:**
- Produces: `func ParseToolCalls(text string) (before string, calls []ToolCall, err error)`.

- [ ] **Step 1: Write the failing test**

```go
package gemma

import (
	"reflect"
	"testing"
)

func TestParseToolCalls(t *testing.T) {
	for _, c := range []struct {
		name   string
		in     string
		before string
		calls  []ToolCall
	}{
		{"none", "just prose", "just prose", nil},
		{"one", `<|tool_call>call:weather{city:<|"|>Lyon<|"|>}<tool_call|>`, "",
			[]ToolCall{{Name: "weather", Arguments: map[string]any{"city": "Lyon"}}}},
		{"after prose", `Let me look.<|tool_call>call:now{}<tool_call|>`, "Let me look.",
			[]ToolCall{{Name: "now", Arguments: map[string]any{}}}},
		{"types", `<|tool_call>call:tags{count:2,pinned:true,ratio:0.5,labels:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>`, "",
			[]ToolCall{{Name: "tags", Arguments: map[string]any{
				"count": float64(2), "pinned": true, "ratio": 0.5,
				"labels": []any{"a", "b"}}}}},
		{"nested", `<|tool_call>call:f{o:{a:1,b:<|"|>x<|"|>}}<tool_call|>`, "",
			[]ToolCall{{Name: "f", Arguments: map[string]any{
				"o": map[string]any{"a": float64(1), "b": "x"}}}}},
		{"two", `<|tool_call>call:a{}<tool_call|><|tool_call>call:b{k:1}<tool_call|>`, "",
			[]ToolCall{{Name: "a", Arguments: map[string]any{}},
				{Name: "b", Arguments: map[string]any{"k": float64(1)}}}},
		{"quote inside a string", `<|tool_call>call:f{s:<|"|>a,b}c<|"|>}<tool_call|>`, "",
			[]ToolCall{{Name: "f", Arguments: map[string]any{"s": "a,b}c"}}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			before, calls, err := ParseToolCalls(c.in)
			if err != nil {
				t.Fatal(err)
			}
			if before != c.before {
				t.Fatalf("before %q, want %q", before, c.before)
			}
			if len(calls) != len(c.calls) {
				t.Fatalf("%d calls, want %d: %#v", len(calls), len(c.calls), calls)
			}
			for i := range calls {
				if calls[i].Name != c.calls[i].Name ||
					!reflect.DeepEqual(calls[i].Arguments, c.calls[i].Arguments) {
					t.Fatalf("call %d is %#v, want %#v", i, calls[i], c.calls[i])
				}
			}
		})
	}
}

func TestParseToolCallsRefusesWhatItCannotRead(t *testing.T) {
	for _, in := range []string{
		`<|tool_call>call:weather{city:<|"|>Lyon<|"|>`,   // never closed
		`<|tool_call>weather{}<tool_call|>`,              // no call: prefix
		`<|tool_call>call:weather{city}<tool_call|>`,     // a key with no value
		`<|tool_call>call:weather{city:<|"|>Lyon}<tool_call|>`, // a string left open
		`<|tool_call>call:{}<tool_call|>`,                // no name
	} {
		if _, _, err := ParseToolCalls(in); err == nil {
			t.Fatalf("%q should be refused rather than half-read", in)
		}
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./gemma/ -run ParseToolCalls -v`
Expected: FAIL — `undefined: ParseToolCalls`.

- [ ] **Step 3: Write `gemma/toolcall.go`**

```go
package gemma

// Reading back what the model wrote.
//
// The arguments of a call are not JSON: <|"|> is the quote, keys are bare, and
// a string may hold a brace or a comma. So this is a scanner rather than a
// call to encoding/json — and it refuses what it cannot read rather than
// returning half a call, because a half-read call would be executed.

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseToolCalls splits an answer into the prose that came before the first
// call and the calls that followed.
func ParseToolCalls(text string) (string, []ToolCall, error) {
	head := strings.Index(text, toolCallOpen)
	if head < 0 {
		return text, nil, nil
	}
	before := text[:head]
	rest := text[head:]

	var calls []ToolCall
	for strings.HasPrefix(rest, toolCallOpen) {
		rest = rest[len(toolCallOpen):]
		if !strings.HasPrefix(rest, "call:") {
			return before, nil, fmt.Errorf("gemma: a tool call that does not open with call:")
		}
		rest = rest[len("call:"):]
		brace := strings.Index(rest, "{")
		if brace < 0 {
			return before, nil, fmt.Errorf("gemma: a tool call with no arguments block")
		}
		name := strings.TrimSpace(rest[:brace])
		if name == "" {
			return before, nil, fmt.Errorf("gemma: a tool call with no function name")
		}
		p := &argScanner{s: rest[brace:]}
		args, err := p.mapping()
		if err != nil {
			return before, nil, err
		}
		rest = p.s[p.at:]
		if !strings.HasPrefix(rest, toolCallClose) {
			return before, nil, fmt.Errorf("gemma: a tool call that never closes")
		}
		rest = rest[len(toolCallClose):]
		calls = append(calls, ToolCall{Name: name, Arguments: args})
	}
	return before, calls, nil
}

// argScanner walks one argument block.
type argScanner struct {
	s  string
	at int
}

func (p *argScanner) mapping() (map[string]any, error) {
	if !p.take("{") {
		return nil, fmt.Errorf("gemma: expected { at %d", p.at)
	}
	out := map[string]any{}
	if p.take("}") {
		return out, nil
	}
	for {
		key, err := p.key()
		if err != nil {
			return nil, err
		}
		if !p.take(":") {
			return nil, fmt.Errorf("gemma: the key %q has no value", key)
		}
		value, err := p.value()
		if err != nil {
			return nil, err
		}
		out[key] = value
		if p.take(",") {
			continue
		}
		if p.take("}") {
			return out, nil
		}
		return nil, fmt.Errorf("gemma: expected , or } at %d", p.at)
	}
}

// key reads a bare key, up to its colon.
func (p *argScanner) key() (string, error) {
	start := p.at
	for p.at < len(p.s) && p.s[p.at] != ':' && p.s[p.at] != ',' && p.s[p.at] != '}' {
		p.at++
	}
	key := strings.TrimSpace(p.s[start:p.at])
	if key == "" {
		return "", fmt.Errorf("gemma: an argument with no key at %d", start)
	}
	return key, nil
}

func (p *argScanner) value() (any, error) {
	switch {
	case p.peek(quote):
		return p.text()
	case p.peek("{"):
		return p.mapping()
	case p.peek("["):
		return p.list()
	default:
		return p.scalar()
	}
}

// text reads a quoted string: everything up to the closing <|"|>, whatever it
// holds.
func (p *argScanner) text() (string, error) {
	p.take(quote)
	end := strings.Index(p.s[p.at:], quote)
	if end < 0 {
		return "", fmt.Errorf("gemma: a string that is never closed")
	}
	out := p.s[p.at : p.at+end]
	p.at += end + len(quote)
	return out, nil
}

func (p *argScanner) list() ([]any, error) {
	p.take("[")
	out := []any{}
	if p.take("]") {
		return out, nil
	}
	for {
		v, err := p.value()
		if err != nil {
			return nil, err
		}
		out = append(out, v)
		if p.take(",") {
			continue
		}
		if p.take("]") {
			return out, nil
		}
		return nil, fmt.Errorf("gemma: expected , or ] at %d", p.at)
	}
}

// scalar reads a bare word: a number, a boolean, or None.
func (p *argScanner) scalar() (any, error) {
	start := p.at
	for p.at < len(p.s) && p.s[p.at] != ',' && p.s[p.at] != '}' && p.s[p.at] != ']' {
		p.at++
	}
	word := strings.TrimSpace(p.s[start:p.at])
	switch word {
	case "true":
		return true, nil
	case "false":
		return false, nil
	case "None", "null":
		return nil, nil
	case "":
		return nil, fmt.Errorf("gemma: an argument with no value at %d", start)
	}
	if f, err := strconv.ParseFloat(word, 64); err == nil {
		return f, nil
	}
	return word, nil
}

func (p *argScanner) peek(what string) bool {
	return strings.HasPrefix(p.s[p.at:], what)
}

func (p *argScanner) take(what string) bool {
	if p.peek(what) {
		p.at += len(what)
		return true
	}
	return false
}
```

- [ ] **Step 4: Run the tests**

Run: `go test ./gemma/ -run ParseToolCalls -v`
Expected: PASS.

- [ ] **Step 5: Run the whole package**

Run: `go test ./gemma/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gemma/toolcall.go gemma/toolcall_test.go
git commit -m "A call the model wrote, read back into arguments"
```

---

### Task 5: A cache that survives stateless requests

**Files:**
- Create: `cmd/serve/context.go`
- Test: `cmd/serve/context_test.go`

This is the heart of the server and is built before the HTTP layer, against the
scripted engine, so its rules are tested without weights.

**Interfaces:**
- Produces:

```go
type Engine interface {
	ForwardBatch(tokens []int32, startPos int) [][]float32
	Logits(hidden, out []float32)
	Reset()
}

type Vocabulary interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Piece(id int32, special bool) string
	IsEOG(id int32) bool
}

func NewContext(e Engine, window, maxContext int, now func() time.Time, ttl time.Duration) *Context
func (c *Context) Prefill(ids []int32) (hidden []float32, fed int, err error)
func (c *Context) Advance(hidden []float32, id int32) []float32
func (c *Context) Pos() int
```

`Prefill` is the whole trick: it compares `ids` with what the cache holds,
feeds only the divergence, and returns the hidden state of the last position
along with how many positions it actually fed — which the test reads to prove
the prefix was reused.

- [ ] **Step 1: Write the failing test**

`cmd/serve/context_test.go` reuses the scripted engine idea from
`cmd/chat/session_test.go`; it is copied rather than shared, because the two
commands do not import one another.

```go
package main

import (
	"testing"
	"time"
)

// An engine that records what it was fed and at which position, and whose
// hidden state is the token itself, so a test can see which state came back.
type recordingEngine struct {
	fed    []int32
	posOf  []int
	resets int
}

func (e *recordingEngine) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	hidden := make([][]float32, len(tokens))
	for i, token := range tokens {
		e.fed = append(e.fed, token)
		e.posOf = append(e.posOf, startPos+i)
		hidden[i] = []float32{float32(token)}
	}
	return hidden
}

func (e *recordingEngine) Logits(hidden, out []float32) {
	for i := range out {
		out[i] = 0
	}
}

func (e *recordingEngine) Reset() { e.resets++ }

func TestPrefillFeedsTheWholePromptTheFirstTime(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(e, 0, 4096, time.Now, 0)
	_, fed, err := c.Prefill([]int32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if fed != 3 || c.Pos() != 3 {
		t.Fatalf("fed %d, pos %d", fed, c.Pos())
	}
}

// The second request repeats the conversation and adds to it: only the
// addition is fed.
func TestPrefillReusesTheCommonPrefix(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(e, 0, 4096, time.Now, 0)
	if _, _, err := c.Prefill([]int32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	hidden, fed, err := c.Prefill([]int32{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	if fed != 2 {
		t.Fatalf("fed %d positions, want 2: the prefix was not reused", fed)
	}
	if hidden[0] != 5 {
		t.Fatalf("the hidden state came from token %v, want the last one", hidden[0])
	}
	if c.Pos() != 5 {
		t.Fatalf("pos %d", c.Pos())
	}
	if e.posOf[3] != 3 || e.posOf[4] != 4 {
		t.Fatalf("positions %v: the continuation did not carry on", e.posOf)
	}
}

// A prompt identical to what is cached still has to feed its last position:
// the hidden state of a position that is already in the cache was not kept.
func TestPrefillOfAnIdenticalPromptFeedsOnePosition(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(e, 0, 4096, time.Now, 0)
	if _, _, err := c.Prefill([]int32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	_, fed, err := c.Prefill([]int32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	if fed != 1 {
		t.Fatalf("fed %d, want the last position only", fed)
	}
	if c.Pos() != 3 {
		t.Fatalf("pos %d", c.Pos())
	}
}

// A different conversation diverges early and is fed from the divergence.
func TestPrefillFeedsFromTheDivergence(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(e, 0, 4096, time.Now, 0)
	if _, _, err := c.Prefill([]int32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	_, fed, err := c.Prefill([]int32{1, 2, 9})
	if err != nil {
		t.Fatal(err)
	}
	if fed != 1 {
		t.Fatalf("fed %d, want the one position that differs", fed)
	}
	if c.Pos() != 3 {
		t.Fatalf("pos %d", c.Pos())
	}
}

// Rewinding a ring: the positions inside the window that the longer run
// overwrote have to be fed again, so prefill restarts a window early.
func TestRewindingAWindowRestartsAWindowEarly(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(e, 4, 4096, time.Now, 0)
	long := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	if _, _, err := c.Prefill(long); err != nil {
		t.Fatal(err)
	}
	short := []int32{1, 2, 3, 4, 5, 9}
	_, fed, err := c.Prefill(short)
	if err != nil {
		t.Fatal(err)
	}
	// The prefix is 5 long, the window is 4: positions 2..5 have to be fed
	// again, which is 4 positions counting the one that differs.
	if fed != 4 {
		t.Fatalf("fed %d, want the window's worth", fed)
	}
	if e.posOf[len(e.posOf)-1] != 5 || c.Pos() != 6 {
		t.Fatalf("positions %v, pos %d", e.posOf, c.Pos())
	}
}

// Appending never rewinds, even with a window.
func TestAppendingWithAWindowFeedsOnlyTheAddition(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(e, 4, 4096, time.Now, 0)
	if _, _, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	_, fed, err := c.Prefill([]int32{1, 2, 3, 4, 5, 6, 7})
	if err != nil {
		t.Fatal(err)
	}
	if fed != 1 {
		t.Fatalf("fed %d, want 1", fed)
	}
}

// The time to live drops what is held, so the next request starts over.
func TestTheTimeToLiveForgetsTheCache(t *testing.T) {
	clock := time.Unix(0, 0)
	e := &recordingEngine{}
	c := NewContext(e, 0, 4096, func() time.Time { return clock }, time.Minute)
	if _, _, err := c.Prefill([]int32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	clock = clock.Add(2 * time.Minute)
	_, fed, err := c.Prefill([]int32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if fed != 4 {
		t.Fatalf("fed %d, want the whole conversation after the cache expired", fed)
	}
	if e.resets != 1 {
		t.Fatalf("%d resets", e.resets)
	}
}

// Inside the delay, nothing is forgotten.
func TestTheTimeToLiveKeepsTheCacheUntilItExpires(t *testing.T) {
	clock := time.Unix(0, 0)
	e := &recordingEngine{}
	c := NewContext(e, 0, 4096, func() time.Time { return clock }, time.Minute)
	if _, _, err := c.Prefill([]int32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	clock = clock.Add(30 * time.Second)
	if _, fed, _ := c.Prefill([]int32{1, 2, 3, 4}); fed != 1 {
		t.Fatalf("fed %d, want 1", fed)
	}
	if e.resets != 0 {
		t.Fatalf("%d resets", e.resets)
	}
}

func TestAPromptPastTheContextIsRefused(t *testing.T) {
	e := &recordingEngine{}
	c := NewContext(e, 0, 4, time.Now, 0)
	if _, _, err := c.Prefill([]int32{1, 2, 3, 4, 5}); err == nil {
		t.Fatal("a prompt past the context should be refused rather than wrap the cache")
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./cmd/serve/ -v`
Expected: FAIL — no such package, then `undefined: NewContext`.

- [ ] **Step 3: Write `cmd/serve/context.go`**

```go
package main

// What the KV cache holds, across requests that carry no state.
//
// /v1/chat/completions is stateless: every request sends the whole
// conversation. The cache is not — it is the whole reason a turn costs one
// turn rather than the conversation. So the server remembers which tokens are
// in the cache, and a request only pays for what its prompt does not share
// with them.
//
// One rule earns its own paragraph. A sliding-window block stores its keys in
// a ring of exactly the window, so writing past position P and then rewinding
// to P overwrites the slots of positions P-W+1 … Q-W, which are still visible
// from P. Rewinding therefore restarts a window early rather than at P.
// Rewriting those positions is idempotent for the global blocks. Appending —
// which is what a conversation growing by one exchange does — rewinds nothing
// and costs nothing.

import (
	"fmt"
	"time"
)

// Engine is the part of gemma.Model the server uses.
type Engine interface {
	ForwardBatch(tokens []int32, startPos int) [][]float32
	Logits(hidden, out []float32)
	Reset()
}

// Vocabulary is the part of bpe.Vocab the server uses.
type Vocabulary interface {
	Encode(text string, addBOS, parseSpecial bool) []int32
	Piece(id int32, special bool) string
	IsEOG(id int32) bool
}

// promptBatch is how many positions go through the model together. Thirty-two
// is what cmd/chat measured: the gain is flat by sixteen, and past sixty-four
// the activations stop fitting in the caches.
const promptBatch = 32

type Context struct {
	engine     Engine
	window     int // the largest sliding window; 0 when every block is global
	maxContext int
	now        func() time.Time
	ttl        time.Duration

	held []int32 // the tokens the cache holds, position by position
	last time.Time
}

func NewContext(e Engine, window, maxContext int, now func() time.Time, ttl time.Duration) *Context {
	return &Context{engine: e, window: window, maxContext: maxContext, now: now, ttl: ttl}
}

// Pos is the position the next token would be fed at.
func (c *Context) Pos() int { return len(c.held) }

// Prefill brings the cache up to ids and returns the hidden state of the last
// position, with the number of positions it had to feed.
func (c *Context) Prefill(ids []int32) ([]float32, int, error) {
	if len(ids) == 0 {
		return nil, 0, fmt.Errorf("serve: an empty prompt")
	}
	if len(ids) > c.maxContext {
		return nil, 0, fmt.Errorf("serve: the conversation is %d positions and the context is %d: start the server with a larger -context, or send less", len(ids), c.maxContext)
	}
	c.expire()

	shared := 0
	for shared < len(c.held) && shared < len(ids) && c.held[shared] == ids[shared] {
		shared++
	}
	// The hidden state of a cached position was not kept, so the last position
	// of the prompt is always fed, whatever is shared.
	from := shared
	if from >= len(ids) {
		from = len(ids) - 1
	}
	// A rewind past what was written corrupts the ring of a window block.
	if len(c.held) > from && c.window > 0 {
		if early := from - c.window + 1; early > 0 {
			from = early
		} else {
			from = 0
		}
	}

	var hidden []float32
	for at := from; at < len(ids); at += promptBatch {
		to := min(at+promptBatch, len(ids))
		states := c.engine.ForwardBatch(ids[at:to], at)
		hidden = states[len(states)-1]
	}
	c.held = append(c.held[:0], ids...)
	c.last = c.now()
	return hidden, len(ids) - from, nil
}

// Advance feeds one drawn token and returns the state it produced.
func (c *Context) Advance(hidden []float32, id int32) []float32 {
	states := c.engine.ForwardBatch([]int32{id}, len(c.held))
	c.held = append(c.held, id)
	c.last = c.now()
	return states[0]
}

// expire drops what is held once the time to live has passed. The memory is
// allocated at startup and is not released here: what expires is the record of
// whose conversation is in it.
func (c *Context) expire() {
	if c.ttl <= 0 || c.held == nil {
		return
	}
	if c.now().Sub(c.last) < c.ttl {
		return
	}
	c.engine.Reset()
	c.held = nil
}
```

Add `cmd/serve/doc.go` holding the package comment for the command, so the
file above stays about the cache.

- [ ] **Step 4: Run the tests**

Run: `go test ./cmd/serve/ -v`
Expected: PASS, all nine.

- [ ] **Step 5: Commit**

```bash
git add cmd/serve/context.go cmd/serve/context_test.go cmd/serve/doc.go
git commit -m "A KV cache reused across requests that carry no state"
```

---

### Task 6: Generating an answer, calls included

**Files:**
- Create: `cmd/serve/generate.go`
- Test: `cmd/serve/generate_test.go`

**Interfaces:**
- Consumes: `Context`, `Vocabulary` from Task 5; `gemma.ParseToolCalls` from Task 4.
- Produces:

```go
type Generator struct{ … }
func NewGenerator(ctx *Context, v Vocabulary, vocabSize int, maxTokens int) *Generator
// Generate draws an answer, handing each piece of prose to emit as it comes.
// A run of text that opens a tool call is withheld until the call closes, so
// emit never sees half a call.
func (g *Generator) Generate(ids []int32, p sample.Params, stop []string, emit func(text string) error) (Answer, error)

type Answer struct {
	Text      string
	ToolCalls []gemma.ToolCall
	Prompt    int
	Generated int
	Reason    string // "stop", "length", "tool_calls"
}
```

- [ ] **Step 1: Write the failing test**

```go
package main

import (
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/sample"
)

// A vocabulary of whole words: each space-separated piece is one token, which
// is enough to script an answer.
type wordVocab struct {
	texts []string
	index map[string]int32
}

func newWordVocab() *wordVocab { return &wordVocab{index: map[string]int32{}} }

func (v *wordVocab) id(text string) int32 {
	if id, ok := v.index[text]; ok {
		return id
	}
	v.texts = append(v.texts, text)
	v.index[text] = int32(len(v.texts) - 1)
	return int32(len(v.texts) - 1)
}

func (v *wordVocab) Encode(text string, addBOS, parseSpecial bool) []int32 {
	var out []int32
	for _, word := range strings.Fields(text) {
		out = append(out, v.id(word))
	}
	return out
}

func (v *wordVocab) Piece(id int32, special bool) string { return v.texts[id] }
func (v *wordVocab) IsEOG(id int32) bool                 { return v.texts[id] == "<turn|>" }

// An engine that speaks a script: each Logits names the next word.
type scriptedEngine struct {
	vocab  *wordVocab
	script []string
	at     int
}

func (e *scriptedEngine) ForwardBatch(tokens []int32, startPos int) [][]float32 {
	hidden := make([][]float32, len(tokens))
	for i := range tokens {
		hidden[i] = []float32{0}
	}
	return hidden
}

func (e *scriptedEngine) Logits(hidden, out []float32) {
	for i := range out {
		out[i] = 0
	}
	word := "<turn|>"
	if e.at < len(e.script) {
		word = e.script[e.at]
	}
	e.at++
	out[e.vocab.id(word)] = 100
}

func (e *scriptedEngine) Reset() {}

func newGenerator(script []string, maxTokens int) (*Generator, *wordVocab) {
	v := newWordVocab()
	e := &scriptedEngine{vocab: v, script: script}
	for _, word := range script {
		v.id(word)
	}
	v.id("<turn|>")
	ctx := NewContext(e, 0, 4096, time.Now, 0)
	return NewGenerator(ctx, v, 4096, maxTokens), v
}

func greedy() sample.Params { return sample.Params{Temperature: 0} }

func TestGenerateStreamsProse(t *testing.T) {
	g, v := newGenerator([]string{"hello", "there", "<turn|>"}, 32)
	var seen []string
	answer, err := g.Generate(v.Encode("a b", false, true), greedy(), nil,
		func(text string) error { seen = append(seen, text); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if answer.Text != "hellothere" || answer.Reason != "stop" {
		t.Fatalf("%+v", answer)
	}
	if strings.Join(seen, "") != "hellothere" || len(seen) != 2 {
		t.Fatalf("streamed %v", seen)
	}
}

func TestGenerateWithholdsAPartialCall(t *testing.T) {
	script := []string{"Looking.", `<|tool_call>call:weather{city:<|"|>Lyon<|"|>}`, "<tool_call|>", "<turn|>"}
	g, v := newGenerator(script, 32)
	var seen []string
	answer, err := g.Generate(v.Encode("a", false, true), greedy(), nil,
		func(text string) error { seen = append(seen, text); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || seen[0] != "Looking." {
		t.Fatalf("streamed %v: a call reached the client as text", seen)
	}
	if len(answer.ToolCalls) != 1 || answer.ToolCalls[0].Name != "weather" {
		t.Fatalf("calls %#v", answer.ToolCalls)
	}
	if answer.ToolCalls[0].Arguments["city"] != "Lyon" {
		t.Fatalf("arguments %#v", answer.ToolCalls[0].Arguments)
	}
	if answer.Reason != "tool_calls" {
		t.Fatalf("reason %q", answer.Reason)
	}
	if answer.Text != "Looking." {
		t.Fatalf("text %q: the call belongs in ToolCalls, not in the content", answer.Text)
	}
	if answer.ToolCalls[0].ID == "" {
		t.Fatal("a call needs an identifier the client can echo back")
	}
}

func TestGenerateStopsOnTheTokenLimit(t *testing.T) {
	g, v := newGenerator([]string{"and", "and", "and", "and"}, 2)
	answer, err := g.Generate(v.Encode("a", false, true), greedy(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer.Generated != 2 || answer.Reason != "length" {
		t.Fatalf("%+v", answer)
	}
}

func TestGenerateStopsOnAStopString(t *testing.T) {
	g, v := newGenerator([]string{"one", "two", "STOP", "three"}, 32)
	answer, err := g.Generate(v.Encode("a", false, true), greedy(), []string{"STOP"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer.Text != "onetwo" || answer.Reason != "stop" {
		t.Fatalf("%+v", answer)
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./cmd/serve/ -run Generate -v`
Expected: FAIL — `undefined: NewGenerator`.

- [ ] **Step 3: Write `cmd/serve/generate.go`**

```go
package main

// Drawing one answer.
//
// Prose reaches the client as it is drawn. A tool call does not: as soon as
// <|tool_call> appears the output is held back until <tool_call|> closes it,
// and the call then leaves as one piece. A client that received half a call
// would have half a function's arguments, and no way to know it.

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/sample"
)

type Generator struct {
	ctx       *Context
	vocab     Vocabulary
	logits    []float32
	maxTokens int
	calls     int // how many calls this server has handed out, for identifiers
}

type Answer struct {
	Text      string
	ToolCalls []gemma.ToolCall
	Prompt    int
	Generated int
	Reason    string // "stop", "length" or "tool_calls"
}

func NewGenerator(ctx *Context, v Vocabulary, vocabSize, maxTokens int) *Generator {
	return &Generator{ctx: ctx, vocab: v, logits: make([]float32, vocabSize), maxTokens: maxTokens}
}

const callOpen = "<|tool_call>"
const callClose = "<tool_call|>"

// Generate draws an answer for a prompt already rendered and encoded. emit,
// when it is not nil, receives each piece of prose as it is drawn.
func (g *Generator) Generate(ids []int32, p sample.Params, stop []string, emit func(string) error) (Answer, error) {
	hidden, fed, err := g.ctx.Prefill(ids)
	if err != nil {
		return Answer{}, err
	}
	answer := Answer{Prompt: fed, Reason: "stop"}
	sampler := sample.New(p)

	var drawn strings.Builder // everything drawn, calls included
	var held strings.Builder  // what is withheld while a call is open
	inCall := false
	sent := 0 // how much prose has left through emit

	for answer.Generated < g.maxTokens {
		g.ctx.engine.Logits(hidden, g.logits)
		id := sampler.Pick(g.logits)
		answer.Generated++
		if g.vocab.IsEOG(id) {
			break
		}
		piece := g.vocab.Piece(id, false)
		drawn.WriteString(piece)

		if index := strings.Index(drawn.String()[sent:], callOpen); !inCall && index >= 0 {
			// The prose before the call still goes out; the rest is held.
			text := drawn.String()[sent : sent+index]
			if text != "" && emit != nil {
				if err := emit(text); err != nil {
					return answer, err
				}
			}
			sent += index
			inCall = true
			held.Reset()
		}
		if inCall {
			held.Reset()
			held.WriteString(drawn.String()[sent:])
			if strings.Contains(held.String(), callClose) {
				inCall = false
				sent = drawn.Len()
			}
		} else if emit != nil {
			if err := emit(piece); err != nil {
				return answer, err
			}
			sent = drawn.Len()
		} else {
			sent = drawn.Len()
		}

		if cut, hit := cutAtStop(drawn.String(), stop); hit {
			text := cut
			answer.Text = text
			answer.Reason = "stop"
			return g.finish(answer, text)
		}
		hidden = g.ctx.Advance(hidden, id)
	}
	if answer.Generated >= g.maxTokens {
		answer.Reason = "length"
	}
	return g.finish(answer, drawn.String())
}

// finish splits what was drawn into prose and calls, and names the calls.
func (g *Generator) finish(answer Answer, drawn string) (Answer, error) {
	before, calls, err := gemma.ParseToolCalls(drawn)
	if err != nil {
		return answer, fmt.Errorf("the model wrote a call this server cannot read: %w", err)
	}
	answer.Text = before
	for i := range calls {
		g.calls++
		calls[i].ID = "call_" + strconv.Itoa(g.calls)
	}
	answer.ToolCalls = calls
	if len(calls) > 0 {
		answer.Reason = "tool_calls"
	}
	return answer, nil
}

// cutAtStop reports whether one of the stop strings has appeared, and returns
// what came before it.
func cutAtStop(text string, stop []string) (string, bool) {
	for _, s := range stop {
		if s == "" {
			continue
		}
		if i := strings.Index(text, s); i >= 0 {
			return text[:i], true
		}
	}
	return text, false
}
```

Note for the implementer: `Generator` reaches `g.ctx.engine` directly for
`Logits`. Add the small method to `Context` rather than exporting the field:

```go
// Logits reads the distribution the last state produced.
func (c *Context) Logits(hidden, out []float32) { c.engine.Logits(hidden, out) }
```

and call `g.ctx.Logits(hidden, g.logits)`.

- [ ] **Step 4: Run the tests until they pass**

Run: `go test ./cmd/serve/ -v`
Expected: PASS. If the withholding test fails by emitting part of a call, the
bug is in the bookkeeping of `sent`, not in the parser.

- [ ] **Step 5: Commit**

```bash
git add cmd/serve/generate.go cmd/serve/context.go cmd/serve/generate_test.go
git commit -m "An answer drawn, with the calls held back until they close"
```

---

### Task 7: The HTTP layer, without streaming

**Files:**
- Create: `cmd/serve/api.go` (the request and response types)
- Create: `cmd/serve/server.go` (the handlers)
- Test: `cmd/serve/server_test.go`

**Interfaces:**
- Consumes: `Generator`, `Answer`, `Context`, `Vocabulary`.
- Produces: `func NewServer(g *Generator, v Vocabulary, name string, emptyThought bool, defaults sample.Params) *Server`, `func (s *Server) Handler() http.Handler`.

- [ ] **Step 1: Write the failing test**

```go
package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newTestServer(script []string) *Server {
	g, v := newGenerator(script, 64)
	return NewServer(g, v, "test-model", false, greedy())
}

func post(t *testing.T, s *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)
	return w
}

func TestModelsListsTheOneModel(t *testing.T) {
	s := newTestServer(nil)
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var out struct {
		Object string `json:"object"`
		Data   []struct {
			ID     string `json:"id"`
			Object string `json:"object"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "list" || len(out.Data) != 1 || out.Data[0].ID != "test-model" {
		t.Fatalf("%s", w.Body)
	}
}

func TestCompletionAnswers(t *testing.T) {
	s := newTestServer([]string{"hello", "<turn|>"})
	w := post(t, s, `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Index        int           `json:"index"`
			Message      gemmaMessage  `json:"message"`
			FinishReason string        `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "chat.completion" {
		t.Fatalf("object %q", out.Object)
	}
	if len(out.Choices) != 1 || out.Choices[0].Message.Content != "hello" {
		t.Fatalf("%s", w.Body)
	}
	if out.Choices[0].Message.Role != "assistant" || out.Choices[0].FinishReason != "stop" {
		t.Fatalf("%s", w.Body)
	}
	if out.Usage.TotalTokens != out.Usage.PromptTokens+out.Usage.CompletionTokens {
		t.Fatalf("usage %+v", out.Usage)
	}
}

// gemmaMessage mirrors what the server writes, so the test reads it back.
type gemmaMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	ToolCalls []json.RawMessage `json:"tool_calls"`
}

func TestCompletionReportsAToolCall(t *testing.T) {
	s := newTestServer([]string{`<|tool_call>call:weather{city:<|"|>Lyon<|"|>}<tool_call|>`, "<turn|>"})
	w := post(t, s, `{"model":"test-model","messages":[{"role":"user","content":"weather?"}],
		"tools":[{"type":"function","function":{"name":"weather","description":"w",
		"parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	c := out.Choices[0]
	if c.FinishReason != "tool_calls" || len(c.Message.ToolCalls) != 1 {
		t.Fatalf("%s", w.Body)
	}
	call := c.Message.ToolCalls[0]
	if call.Type != "function" || call.Function.Name != "weather" {
		t.Fatalf("%s", w.Body)
	}
	if call.Function.Arguments != `{"city":"Lyon"}` {
		t.Fatalf("arguments %q", call.Function.Arguments)
	}
	if call.ID == "" {
		t.Fatal("a call with no identifier cannot be answered")
	}
}

func TestCompletionRefusesWhatItDoesNotImplement(t *testing.T) {
	s := newTestServer([]string{"x", "<turn|>"})
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi"}],"n":2}`,
		`{"messages":[{"role":"user","content":"hi"}],"logprobs":true}`,
		`{"messages":[{"role":"user","content":"hi"}],"tool_choice":"required"}`,
		`{"messages":[]}`,
		`{`,
	} {
		w := post(t, s, body)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("%s answered %d, want 400", body, w.Code)
		}
		var out struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
			} `json:"error"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil || out.Error.Message == "" {
			t.Fatalf("the error envelope was %s", w.Body)
		}
	}
}

func TestAnUnknownPathIs404(t *testing.T) {
	s := newTestServer(nil)
	req := httptest.NewRequest("GET", "/v1/embeddings", nil)
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		t.Fatalf("%d", w.Code)
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./cmd/serve/ -run 'Completion|Models|Unknown' -v`
Expected: FAIL — `undefined: NewServer`.

- [ ] **Step 3: Write `cmd/serve/api.go`**

```go
package main

// The shapes OpenAI's API defines, as much of them as this server implements.

import "github.com/ThiraSoft/golem/gemma"

type completionRequest struct {
	Model       string          `json:"model"`
	Messages    []gemma.Message `json:"messages"`
	Tools       []gemma.Tool    `json:"tools"`
	ToolChoice  any             `json:"tool_choice"`
	Stream      bool            `json:"stream"`
	Temperature *float64        `json:"temperature"`
	TopP        *float64        `json:"top_p"`
	TopK        *int            `json:"top_k"`
	MaxTokens   *int            `json:"max_tokens"`
	Seed        *uint64         `json:"seed"`
	Stop        stopStrings     `json:"stop"`
	N           *int            `json:"n"`
	LogProbs    *bool           `json:"logprobs"`
}

// stopStrings reads the two spellings the API allows: one string, or a list.
type stopStrings []string

func (s *stopStrings) UnmarshalJSON(raw []byte) error {
	if len(raw) > 0 && raw[0] == '"' {
		var one string
		if err := json.Unmarshal(raw, &one); err != nil {
			return err
		}
		*s = []string{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return err
	}
	*s = many
	return nil
}

type responseMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []gemma.ToolCall `json:"tool_calls,omitempty"`
}

type choice struct {
	Index        int              `json:"index"`
	Message      *responseMessage `json:"message,omitempty"`
	Delta        *responseMessage `json:"delta,omitempty"`
	FinishReason *string          `json:"finish_reason"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type completionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   *usage   `json:"usage,omitempty"`
}

type apiError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}
```

(add `"encoding/json"` to the imports).

- [ ] **Step 4: Write `cmd/serve/server.go`**

```go
package main

// The HTTP layer. It knows the shapes of OpenAI's API and nothing about
// attention.
//
// One model and one cache mean one request at a time: a mutex serializes them
// and a second request waits. There is no queue and no batching between
// requests, which the README says plainly rather than implying otherwise.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/sample"
)

type Server struct {
	mu           sync.Mutex
	gen          *Generator
	vocab        Vocabulary
	name         string
	emptyThought bool
	defaults     sample.Params
	served       int
}

func NewServer(g *Generator, v Vocabulary, name string, emptyThought bool, defaults sample.Params) *Server {
	return &Server{gen: g, vocab: v, name: name, emptyThought: emptyThought, defaults: defaults}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("POST /v1/chat/completions", s.completions)
	return mux
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id": s.name, "object": "model", "created": time.Now().Unix(),
			"owned_by": "golem",
		}},
	})
}

func (s *Server) completions(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid_request_error", "the request body is not the JSON this endpoint reads: "+err.Error())
		return
	}
	if err := check(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	prompt, err := gemma.RenderChat(req.Messages, gemma.ChatOptions{
		Tools:               req.Tools,
		EmptyThought:        s.emptyThought,
		AddGenerationPrompt: true,
	})
	if err != nil {
		fail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.served++
	id := "chatcmpl-" + strconv.Itoa(s.served)

	ids := s.vocab.Encode(prompt, false, true)
	params := s.sampling(&req)
	if req.Stream {
		s.stream(w, id, ids, params, req.Stop)
		return
	}
	answer, err := s.gen.Generate(ids, params, req.Stop, nil)
	if err != nil {
		fail(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	reason := answer.Reason
	writeJSON(w, http.StatusOK, completionResponse{
		ID: id, Object: "chat.completion", Created: time.Now().Unix(), Model: s.name,
		Choices: []choice{{
			Index: 0,
			Message: &responseMessage{Role: "assistant", Content: answer.Text,
				ToolCalls: answer.ToolCalls},
			FinishReason: &reason,
		}},
		Usage: &usage{PromptTokens: answer.Prompt, CompletionTokens: answer.Generated,
			TotalTokens: answer.Prompt + answer.Generated},
	})
}

// check refuses what would change the shape of the answer, rather than
// answering something the client did not ask for.
func check(req *completionRequest) error {
	if len(req.Messages) == 0 {
		return fmt.Errorf("a conversation with no message")
	}
	if req.N != nil && *req.N != 1 {
		return fmt.Errorf("n is %d: this server draws one answer", *req.N)
	}
	if req.LogProbs != nil && *req.LogProbs {
		return fmt.Errorf("logprobs is not implemented")
	}
	switch choice := req.ToolChoice.(type) {
	case nil:
	case string:
		if choice != "auto" && choice != "none" {
			return fmt.Errorf("tool_choice %q is not implemented; auto and none are", choice)
		}
	default:
		return fmt.Errorf("tool_choice as an object is not implemented; auto and none are")
	}
	if s, ok := req.ToolChoice.(string); ok && s == "none" {
		req.Tools = nil
	}
	return nil
}

// sampling starts from what the file asks for and lets the request override.
func (s *Server) sampling(req *completionRequest) sample.Params {
	p := s.defaults
	if req.Temperature != nil {
		p.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		p.TopP = float32(*req.TopP)
	}
	if req.TopK != nil {
		p.TopK = *req.TopK
	}
	if req.Seed != nil {
		p.Seed = *req.Seed
	}
	return p
}

func writeJSON(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(body)
}

func fail(w http.ResponseWriter, code int, kind, message string) {
	var body apiError
	body.Error.Message = message
	body.Error.Type = kind
	writeJSON(w, code, body)
}
```

`s.stream` arrives in Task 8. Until then, write it as a method that answers
`fail(w, http.StatusBadRequest, "invalid_request_error", "streaming is not implemented yet")`, so this task compiles and its tests are honest.

The `-max-tokens` per request: `req.MaxTokens` overrides the generator's limit.
Add `func (g *Generator) WithMaxTokens(n int) *Generator` returning a shallow
copy sharing the context, and use it when `req.MaxTokens != nil`.

- [ ] **Step 5: Run the tests**

Run: `go test ./cmd/serve/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cmd/serve/api.go cmd/serve/server.go cmd/serve/server_test.go
git commit -m "An OpenAI-shaped endpoint over one mapped model"
```

---

### Task 8: Streaming

**Files:**
- Create: `cmd/serve/stream.go`
- Modify: `cmd/serve/server.go` (remove the placeholder `stream` method)
- Test: `cmd/serve/stream_test.go`

**Interfaces:**
- Consumes: `Generator.Generate`'s `emit` callback.
- Produces: `func (s *Server) stream(w http.ResponseWriter, id string, ids []int32, p sample.Params, stop []string)`.

The frame sequence: a first chunk carrying `{"role":"assistant"}` in `delta`,
then one chunk per piece of prose, then — if there are calls — one chunk
carrying the whole `tool_calls` array with an `index` on each, then a chunk
with an empty delta and the `finish_reason`, then `data: [DONE]`.

- [ ] **Step 1: Write the failing test**

```go
package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

// frames splits an SSE body into the JSON payloads it carried.
func frames(t *testing.T, body string) []string {
	t.Helper()
	var out []string
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, "data: ") {
			out = append(out, strings.TrimPrefix(line, "data: "))
		}
	}
	if len(out) == 0 {
		t.Fatalf("no SSE frames in %q", body)
	}
	return out
}

func TestStreamSendsProseThenAFinishReasonThenDone(t *testing.T) {
	s := newTestServer([]string{"one", "two", "<turn|>"})
	w := post(t, s, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if got := w.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("content type %q", got)
	}
	f := frames(t, w.Body.String())
	if f[len(f)-1] != "[DONE]" {
		t.Fatalf("the stream did not close with [DONE]: %v", f)
	}
	var text strings.Builder
	var reason string
	for _, frame := range f[:len(f)-1] {
		var chunk struct {
			Object  string `json:"object"`
			Choices []struct {
				Delta struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			t.Fatalf("%q: %v", frame, err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Fatalf("object %q", chunk.Object)
		}
		text.WriteString(chunk.Choices[0].Delta.Content)
		if chunk.Choices[0].FinishReason != nil {
			reason = *chunk.Choices[0].FinishReason
		}
	}
	if text.String() != "onetwo" {
		t.Fatalf("streamed %q", text.String())
	}
	if reason != "stop" {
		t.Fatalf("reason %q", reason)
	}
}

func TestStreamSendsACallInOnePiece(t *testing.T) {
	s := newTestServer([]string{"Looking.", `<|tool_call>call:weather{city:<|"|>Lyon<|"|>}`, "<tool_call|>", "<turn|>"})
	w := post(t, s, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	f := frames(t, w.Body.String())

	var prose strings.Builder
	callFrames := 0
	var reason string
	for _, frame := range f[:len(f)-1] {
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			t.Fatal(err)
		}
		d := chunk.Choices[0].Delta
		prose.WriteString(d.Content)
		if len(d.ToolCalls) > 0 {
			callFrames++
			if d.ToolCalls[0].Function.Arguments != `{"city":"Lyon"}` {
				t.Fatalf("arguments %q left in pieces", d.ToolCalls[0].Function.Arguments)
			}
			if d.ToolCalls[0].ID == "" {
				t.Fatal("a streamed call needs its identifier")
			}
		}
		if chunk.Choices[0].FinishReason != nil {
			reason = *chunk.Choices[0].FinishReason
		}
	}
	if callFrames != 1 {
		t.Fatalf("%d frames carried the call, want exactly one", callFrames)
	}
	if strings.Contains(prose.String(), "tool_call") {
		t.Fatalf("the call leaked into the prose: %q", prose.String())
	}
	if prose.String() != "Looking." {
		t.Fatalf("prose %q", prose.String())
	}
	if reason != "tool_calls" {
		t.Fatalf("reason %q", reason)
	}
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `go test ./cmd/serve/ -run Stream -v`
Expected: FAIL — the placeholder answers 400.

- [ ] **Step 3: Write `cmd/serve/stream.go`**

```go
package main

// Server-sent events.
//
// Prose leaves as it is drawn. A call leaves once, whole, in a single delta:
// OpenAI's API allows a call to arrive in fragments, and clients do reassemble
// them, but a fragment of a function's arguments is a thing that cannot be
// checked, and there is nothing to gain here by sending one.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/ThiraSoft/golem/sample"
)

func (s *Server) stream(w http.ResponseWriter, id string, ids []int32, p sample.Params, stop []string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		fail(w, http.StatusInternalServerError, "server_error", "this connection cannot be streamed to")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	created := time.Now().Unix()
	send := func(c choice) error {
		body, err := json.Marshal(completionResponse{
			ID: id, Object: "chat.completion.chunk", Created: created,
			Model: s.name, Choices: []choice{c},
		})
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", body); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	if err := send(choice{Delta: &responseMessage{Role: "assistant"}}); err != nil {
		return
	}
	answer, err := s.gen.Generate(ids, p, stop, func(text string) error {
		return send(choice{Delta: &responseMessage{Content: text}})
	})
	if err != nil {
		// The header is already out, so the error goes down the stream, which
		// is the only place left for it.
		send(choice{Delta: &responseMessage{Content: "\n[golem: " + err.Error() + "]"}})
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	if len(answer.ToolCalls) > 0 {
		if err := send(choice{Delta: &responseMessage{ToolCalls: answer.ToolCalls}}); err != nil {
			return
		}
	}
	reason := answer.Reason
	send(choice{Delta: &responseMessage{}, FinishReason: &reason})
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}
```

A streamed call needs its `index`, which `gemma.ToolCall.MarshalJSON` writes
only when it is set. Set it in `stream` before sending:

```go
	for i := range answer.ToolCalls {
		answer.ToolCalls[i].Index = i
	}
```

and add to `gemma.ToolCall`:

```go
	// Index numbers a call inside one streamed delta. Zero means unset, which
	// is why it is a pointer on the wire and an int here plus a flag.
	Index    int
	HasIndex bool
```

writing `w.Index = &c.Index` in `MarshalJSON` when `HasIndex` is true. Set
`HasIndex` in `stream`.

- [ ] **Step 4: Run the tests**

Run: `go test ./cmd/serve/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cmd/serve/stream.go cmd/serve/server.go cmd/serve/stream_test.go gemma/tools.go
git commit -m "The answer, streamed, with a call sent whole"
```

---

### Task 9: The command, and what it says about itself

**Files:**
- Create: `cmd/serve/main.go`
- Create: `cmd/serve/README.md`
- Modify: `README.md` (the commands table)
- Create: `cmd/serve/live_test.go` (guarded by `GOLEM_MODEL`)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write `cmd/serve/main.go`**

```go
// Command serve answers an OpenAI-compatible API over a Gemma 4 GGUF, on the
// CPU.
//
//	serve -model gemma-4-E2B-it-QAT-Q4_0.gguf -addr :8080
//
// It implements /v1/chat/completions, streamed or not, tool declarations
// included, and /v1/models. It reports the calls the model makes; running them
// is the client's part, as the protocol has it.
package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/token/bpe"
)

func main() {
	model := flag.String("model", os.Getenv("GOLEM_MODEL"), "Gemma 4 GGUF file (or GOLEM_MODEL)")
	addr := flag.String("addr", "127.0.0.1:8080", "address to listen on")
	context := flag.Int("context", 4096, "positions to keep; the file declares 131072, which no machine here would survive")
	maxTokens := flag.Int("n", 1024, "most tokens to draw for one answer, when the request names no limit")
	ttl := flag.Duration("cache-ttl", 0, "forget a conversation's tokens after this long idle; 0 never forgets. The memory is allocated at startup and is not released either way")
	flag.Parse()

	if *model == "" {
		fail(fmt.Errorf("no model: pass -model, or set GOLEM_MODEL"))
	}
	start := time.Now()
	m, err := gemma.Open(*model, *context)
	if err != nil {
		fail(err)
	}
	defer m.Close()
	vocab, err := bpe.Load(m.File())
	if err != nil {
		fail(err)
	}

	window := 0
	for _, b := range m.Cfg.Blocks {
		if b.Window && b.WindowSize > window {
			window = b.WindowSize
		}
	}
	params := m.Cfg.Sampling
	params.Seed = rand.Uint64()

	ctx := NewContext(m, window, *context, time.Now, *ttl)
	gen := NewGenerator(ctx, vocab, m.Cfg.Vocab, *maxTokens)
	name := strings.TrimSuffix(filepath.Base(*model), ".gguf")
	server := NewServer(gen, vocab, name, m.Cfg.EmptyThought, params)

	fmt.Fprintf(os.Stderr, "%s: %d blocks, %d positions, loaded in %s on %d cores\n",
		name, len(m.Cfg.Blocks), *context, time.Since(start).Round(time.Millisecond), runtime.NumCPU())
	fmt.Fprintf(os.Stderr, "listening on http://%s/v1 — one request at a time\n", *addr)
	if err := http.ListenAndServe(*addr, server.Handler()); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "serve:", err)
	os.Exit(1)
}
```

`gemma.Model` needs a `Reset` method for the `Engine` interface — it already
has one. Check that `*gemma.Model` satisfies `Engine` at compile time with a
line in `context.go`:

```go
var _ Engine = (*gemma.Model)(nil)
```

- [ ] **Step 2: Build it**

Run: `go build ./... && go vet ./...`
Expected: no output.

- [ ] **Step 3: Write the live test**

```go
package main

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/token/bpe"
)

// The one test with weights: a conversation declaring a tool, whose answer is
// a call this server can read. Skipped when the file is not on the machine.
func TestLiveToolCall(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("GOLEM_MODEL is not set")
	}
	m, err := gemma.Open(path, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	vocab, err := bpe.Load(m.File())
	if err != nil {
		t.Fatal(err)
	}
	window := 0
	for _, b := range m.Cfg.Blocks {
		if b.Window && b.WindowSize > window {
			window = b.WindowSize
		}
	}
	params := m.Cfg.Sampling
	params.Temperature = 0
	ctx := NewContext(m, window, 2048, time.Now, 0)
	server := NewServer(NewGenerator(ctx, vocab, m.Cfg.Vocab, 128), vocab,
		"live", m.Cfg.EmptyThought, params)

	body := `{"messages":[{"role":"user","content":"What is the weather in Lyon right now? Use the tool."}],
		"tools":[{"type":"function","function":{"name":"get_weather",
		"description":"Current weather in a city.",
		"parameters":{"type":"object","properties":{
			"city":{"type":"string","description":"The city."}},
		"required":["city"]}}}],"temperature":0}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.Handler().ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content   string           `json:"content"`
				ToolCalls []gemma.ToolCall `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	c := out.Choices[0]
	if c.FinishReason != "tool_calls" || len(c.Message.ToolCalls) == 0 {
		t.Fatalf("the model answered %q with reason %q instead of calling the tool",
			c.Message.Content, c.FinishReason)
	}
	if c.Message.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("it called %q", c.Message.ToolCalls[0].Name)
	}
	t.Logf("arguments: %#v", c.Message.ToolCalls[0].Arguments)
}
```

- [ ] **Step 4: Run everything**

Run: `go test ./...`
Expected: PASS, the live test included when `GOLEM_MODEL` is set. If the model
answers in prose rather than calling, the prompt is the suspect — say so in the
failure and check by hand with `llama-server` before touching the renderer.

- [ ] **Step 5: Write `cmd/serve/README.md`**

Cover: what it serves and on which paths; that it declares tools and reports
calls but runs none; that requests are served one at a time, with no queue and
no cross-request batching; how the prefix of the KV cache is reused and what
`-cache-ttl` does and does not do (it frees no memory); what is refused with a
400 and why (`n`, `logprobs`, `tool_choice` beyond auto and none); and a
`curl` example of a tool call, with the answer it produces.

- [ ] **Step 6: Add the command to the top-level README**

Add a row to the commands table:

```markdown
| [`cmd/serve`](cmd/serve/) | an OpenAI-compatible API over the same weights, tool calls included |
```

- [ ] **Step 7: Commit**

```bash
git add cmd/serve README.md
git commit -m "A server, and what it does not promise"
```

---

## Self-Review

**Spec coverage.** Template outward → Tasks 2 and 3. Template inward → Task 4.
Server, endpoints, streaming, refusals, error envelope → Tasks 7 and 8.
Prefix reuse and the window rewind → Task 5. TTL → Task 5 (`Context.expire`),
surfaced as a flag in Task 9. Concurrency → the mutex in Task 7, stated in the
README in Task 9. Testing → each task carries its own; the live test is Task 9.
Out of scope items appear nowhere, which is the point.

**Types.** `Engine`, `Vocabulary`, `Context`, `Generator`, `Answer`, `Server`
are each defined once and used with the same names afterwards. `gemma.Tool`,
`gemma.Schema`, `gemma.ToolCall` come from Task 1 and every later use matches
those fields. `ToolCall.Index`/`HasIndex` are added in Task 8, which is the
only task that needs them.

**Known sequencing risk.** Task 3 cannot be finished without the fixture from
Task 2, and Task 2 needs the two GGUFs on the machine. If a model file has
moved, stop and say so rather than hand-writing a fixture.
