package gemma

import (
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/chat"
)

// The commands hold a chat.Template and nothing narrower.
var _ chat.Template = (*Template)(nil)

func TestTemplateRendersThroughTheConfig(t *testing.T) {
	msgs := []chat.Message{{Role: "user", Content: "hi"}}
	opt := chat.Options{AddGenerationPrompt: true}

	plain := NewTemplate(&Config{EmptyThought: false})
	got, err := plain.Render(msgs, opt)
	if err != nil {
		t.Fatal(err)
	}
	if want := "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"; got != want {
		t.Fatalf("rendered %q, want %q", got, want)
	}

	// The 12B's template closes the generation prompt with a thought channel
	// opened and closed at once. The config says so; the options do not.
	thought := NewTemplate(&Config{EmptyThought: true})
	got, err = thought.Render(msgs, opt)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(got, "<|turn>model\n<|channel>thought\n<channel|>") {
		t.Fatalf("rendered %q, want the empty thought channel at the end", got)
	}
}

func TestTemplateCallOpenAndParse(t *testing.T) {
	tpl := NewTemplate(&Config{})
	if tpl.CallOpen() != "<|tool_call>" {
		t.Fatalf("CallOpen is %q", tpl.CallOpen())
	}
	before, calls, err := tpl.ParseCalls(`Looking.<|tool_call>call:weather{city:<|"|>Lyon<|"|>}<tool_call|>`)
	if err != nil {
		t.Fatal(err)
	}
	if before != "Looking." || len(calls) != 1 || calls[0].Name != "weather" ||
		calls[0].Arguments["city"] != "Lyon" {
		t.Fatalf("before %q calls %+v", before, calls)
	}
}
