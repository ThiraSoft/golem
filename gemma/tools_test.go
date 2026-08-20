package gemma

import (
	"encoding/json"
	"reflect"
	"strings"
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
	if err := json.Unmarshal([]byte(`{"type":"retrieval"}`), &tool); err == nil {
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
		Index    *int   `json:"index"`
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
	if back.Index != nil {
		t.Fatalf("an index nobody set was written: %s", raw)
	}
}

// Arguments with no key are an empty object, not null: a client feeding the
// call back must get valid JSON.
func TestToolCallWithNoArgumentsMarshalsAsAnEmptyObject(t *testing.T) {
	raw, _ := json.Marshal(ToolCall{ID: "c", Name: "now"})
	if want := `"arguments":"{}"`; !strings.Contains(string(raw), want) {
		t.Fatalf("%s does not hold %s", raw, want)
	}
}

// A streamed call carries its index, zero included.
func TestToolCallWritesTheIndexWhenItIsMeant(t *testing.T) {
	raw, _ := json.Marshal(ToolCall{ID: "c", Name: "now", Index: 0, HasIndex: true})
	if !strings.Contains(string(raw), `"index":0`) {
		t.Fatalf("%s", raw)
	}
}
