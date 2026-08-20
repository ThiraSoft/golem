package gemma

import (
	"reflect"
	"strings"
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
		`<|tool_call>call:weather{city:<|"|>Lyon<|"|>`,         // never closed
		`<|tool_call>weather{}<tool_call|>`,                    // no call: prefix
		`<|tool_call>call:weather{city}<tool_call|>`,           // a key with no value
		`<|tool_call>call:weather{city:<|"|>Lyon}<tool_call|>`, // a string left open
		`<|tool_call>call:{}<tool_call|>`,                      // no name
	} {
		if _, _, err := ParseToolCalls(in); err == nil {
			t.Fatalf("%q should be refused rather than half-read", in)
		}
	}
}

// What the renderer writes, the scanner reads.
func TestACallSurvivesTheRoundTrip(t *testing.T) {
	call := ToolCall{Name: "tags", Arguments: map[string]any{
		"labels": []any{"a", "b"}, "pinned": true, "count": float64(2),
		"note": "one, two}", "nested": map[string]any{"k": "v"}}}
	var b strings.Builder
	writeCall(&b, call)
	_, back, err := ParseToolCalls(b.String())
	if err != nil {
		t.Fatal(err)
	}
	if len(back) != 1 || !reflect.DeepEqual(back[0].Arguments, call.Arguments) {
		t.Fatalf("%#v came back as %#v", call.Arguments, back)
	}
}
