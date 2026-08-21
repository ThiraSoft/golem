package qwen

import (
	"testing"

	"github.com/ThiraSoft/golem/chat"
)

// The commands hold a chat.Template and nothing narrower.
var _ chat.Template = (*Template)(nil)

func TestTemplateRendersAndParses(t *testing.T) {
	tpl := NewTemplate(&Config{})
	got, err := tpl.Render([]chat.Message{{Role: "user", Content: "hi"}},
		chat.Options{AddGenerationPrompt: true})
	if err != nil {
		t.Fatal(err)
	}
	if want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"; got != want {
		t.Fatalf("rendered %q, want %q", got, want)
	}
	if tpl.CallOpen() != "<tool_call>" {
		t.Fatalf("CallOpen is %q", tpl.CallOpen())
	}
	_, calls, err := tpl.ParseCalls("<tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>")
	if err != nil || len(calls) != 1 || calls[0].Name != "now" {
		t.Fatalf("%+v %v", calls, err)
	}
}
