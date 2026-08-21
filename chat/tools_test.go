package chat

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToolReadsTheFunctionWrapper(t *testing.T) {
	var tool Tool
	raw := `{"type":"function","function":{"name":"weather",
		"description":"Today's weather.",
		"parameters":{"type":"object","properties":{"city":{"type":"string"}},
		"required":["city"]}}}`
	if err := json.Unmarshal([]byte(raw), &tool); err != nil {
		t.Fatal(err)
	}
	if tool.Name != "weather" || tool.Description != "Today's weather." {
		t.Fatalf("%+v", tool)
	}
	if tool.Parameters == nil || tool.Parameters.Properties["city"].Type != "string" {
		t.Fatalf("parameters %+v", tool.Parameters)
	}
}

func TestToolRefusesWhatIsNotAFunction(t *testing.T) {
	var tool Tool
	if err := json.Unmarshal([]byte(`{"type":"retrieval","function":{"name":"x"}}`), &tool); err == nil {
		t.Fatal("a tool that is not a function should be refused")
	}
	if err := json.Unmarshal([]byte(`{"type":"function","function":{}}`), &tool); err == nil {
		t.Fatal("a function with no name should be refused")
	}
}

// On the wire arguments are a string holding JSON, because that is what
// OpenAI's API settled on. A call reads both spellings and always writes the
// string, which is what a client feeding the call back expects.
func TestToolCallReadsBothSpellingsOfArguments(t *testing.T) {
	for _, raw := range []string{
		`{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Lyon"}}}`,
		`{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Lyon\"}"}}`,
	} {
		var call ToolCall
		if err := json.Unmarshal([]byte(raw), &call); err != nil {
			t.Fatal(err)
		}
		if call.ID != "call_1" || call.Name != "weather" || call.Arguments["city"] != "Lyon" {
			t.Fatalf("%q gave %+v", raw, call)
		}
	}
}

func TestToolCallWritesArgumentsAsAString(t *testing.T) {
	out, err := json.Marshal(ToolCall{ID: "call_1", Name: "weather",
		Arguments: map[string]any{"city": "Lyon"}})
	if err != nil {
		t.Fatal(err)
	}
	var back struct {
		Function struct {
			Arguments any `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatal(err)
	}
	if _, ok := back.Function.Arguments.(string); !ok {
		t.Fatalf("arguments went out as %T, want a string: %s", back.Function.Arguments, out)
	}
}

func TestToolCallOmitsAnIndexNobodyMeant(t *testing.T) {
	out, _ := json.Marshal(ToolCall{Name: "now"})
	if strings.Contains(string(out), `"index"`) {
		t.Fatalf("an index nobody set was written: %s", out)
	}
	out, _ = json.Marshal(ToolCall{Name: "now", Index: 0, HasIndex: true})
	if !strings.Contains(string(out), `"index":0`) {
		t.Fatalf("index zero is a real index and should be written: %s", out)
	}
}
