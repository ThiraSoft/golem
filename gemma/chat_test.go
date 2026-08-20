package gemma

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type chatCase struct {
	Name                string    `json:"name"`
	Messages            []Message `json:"messages"`
	Tools               []Tool    `json:"tools"`
	EnableThinking      bool      `json:"enable_thinking"`
	AddGenerationPrompt bool      `json:"add_generation_prompt"`
	Rendered            string    `json:"rendered"`
}

// The fixture also carries the one rule the two checkpoints spell differently,
// read from the template it was rendered from.
func loadChatCases(t *testing.T, dir string) ([]chatCase, bool) {
	t.Helper()
	path := filepath.Join(repoRoot(t), "testdata", "gemma", dir, "cases.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("the chat fixture is committed and should be readable: %v", err)
	}
	var file struct {
		EmptyThought bool       `json:"empty_thought"`
		Cases        []chatCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("the fixture holds no cases")
	}
	return file.Cases, file.EmptyThought
}

// The whole point: what Jinja made of Gemma's own template, character for
// character, without a Jinja interpreter.
func TestRenderChatMatchesTheTemplate(t *testing.T) {
	// One fixture per checkpoint, because the two templates are not the same
	// text: chat came from E2B, chat12 from the 12B.
	for _, dir := range []string{"chat", "chat12"} {
		cases, emptyThought := loadChatCases(t, dir)
		for _, c := range cases {
			t.Run(dir+"/"+c.Name, func(t *testing.T) {
				got, err := RenderChat(c.Messages, ChatOptions{
					Tools:               c.Tools,
					EnableThinking:      c.EnableThinking,
					EmptyThought:        emptyThought,
					AddGenerationPrompt: c.AddGenerationPrompt,
				})
				if err != nil {
					t.Fatal(err)
				}
				if got != c.Rendered {
					t.Fatalf("rendered\n%q\nwant\n%q", got, c.Rendered)
				}
			})
		}
	}
}

func TestRenderChatRejectsWhatItCannotRender(t *testing.T) {
	if _, err := RenderChat(nil, ChatOptions{}); err == nil {
		t.Fatal("an empty conversation has no first message and should be refused")
	}
	if _, err := RenderChat([]Message{{Role: "tool", Name: "weather", Content: "x"}}, ChatOptions{}); err == nil {
		t.Fatal("a tool result with no call before it should be refused")
	}
	if _, err := RenderChat([]Message{
		{Role: "user", Content: "one"},
		{Role: "tool", Name: "weather", Content: "x"},
	}, ChatOptions{}); err == nil {
		t.Fatal("a tool result answering a message that made no call should be refused")
	}
	if _, err := RenderChat([]Message{
		{Role: "user", Content: "one"},
		{Role: "system", Content: "late"},
	}, ChatOptions{}); err == nil {
		t.Fatal("a system message after the first is not the template's system turn")
	}
}

func TestStripThinking(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"plain", "plain"},
		{"before<|channel>thought\nhidden<channel|>after", "beforeafter"},
		{"kept<|channel>dropped", "kept"},
		{"  <|channel>x<channel|>  answer  ", "answer"},
		{"a<channel|>b", "ab"},
	} {
		if got := StripThinking(c.in); got != c.want {
			t.Fatalf("StripThinking(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
