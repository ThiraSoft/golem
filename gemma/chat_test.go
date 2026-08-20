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
	EnableThinking      bool      `json:"enable_thinking"`
	AddGenerationPrompt bool      `json:"add_generation_prompt"`
	Rendered            string    `json:"rendered"`
}

func loadChatCases(t *testing.T) []chatCase {
	t.Helper()
	path := filepath.Join(repoRoot(t), "testdata", "gemma", "chat", "cases.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("the chat fixture is committed and should be readable: %v", err)
	}
	var file struct {
		Cases []chatCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("the fixture holds no cases")
	}
	return file.Cases
}

// The whole point: what Jinja made of Gemma's own template, character for
// character, without a Jinja interpreter.
func TestRenderChatMatchesTheTemplate(t *testing.T) {
	for _, c := range loadChatCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			got, err := RenderChat(c.Messages, ChatOptions{
				EnableThinking:      c.EnableThinking,
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

func TestRenderChatRejectsWhatItCannotRender(t *testing.T) {
	if _, err := RenderChat(nil, ChatOptions{}); err == nil {
		t.Fatal("an empty conversation has no first message and should be refused")
	}
	if _, err := RenderChat([]Message{{Role: "tool", Content: "x"}}, ChatOptions{}); err == nil {
		t.Fatal("the tool role is not implemented and should be refused rather than mis-rendered")
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
