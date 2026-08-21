package qwen

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/chat"
)

type chatCase struct {
	Name                string         `json:"name"`
	Messages            []chat.Message `json:"messages"`
	Tools               []chat.Tool    `json:"tools"`
	EnableThinking      bool           `json:"enable_thinking"`
	AddGenerationPrompt bool           `json:"add_generation_prompt"`
	Rendered            string         `json:"rendered"`
}

func loadChatCases(t *testing.T) []chatCase {
	t.Helper()
	path := filepath.Join(repoRoot(t), "testdata", "qwen", "chat", "cases.json")
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

// The whole point: what Jinja made of Qwen3's own template, character for
// character, without a Jinja interpreter.
func TestRenderChatMatchesTheTemplate(t *testing.T) {
	for _, c := range loadChatCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			got, err := RenderChat(c.Messages, chat.Options{
				Tools:               c.Tools,
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

func TestRenderChatRefusesWhatItCannotRender(t *testing.T) {
	if _, err := RenderChat(nil, chat.Options{}); err == nil {
		t.Fatal("an empty conversation should be refused")
	}
	if _, err := RenderChat([]chat.Message{{Role: "tool", Name: "weather", Content: "x"}},
		chat.Options{}); err == nil {
		t.Fatal("a tool result with no call before it should be refused")
	}
	if _, err := RenderChat([]chat.Message{{Role: "narrator", Content: "x"}},
		chat.Options{}); err == nil {
		t.Fatal("an unknown role should be refused rather than approximated")
	}
}

// The template has no developer role and would drop it in silence. Dropping a
// message is worse than either alternative, and refusing it would make an
// OpenAI-compatible server answer differently depending on which engine it
// opened, so it is read as the system message it stands for.
func TestRenderChatReadsDeveloperAsSystem(t *testing.T) {
	as := func(role string) string {
		out, err := RenderChat([]chat.Message{
			{Role: role, Content: "You are terse."},
			{Role: "user", Content: "hi"},
		}, chat.Options{AddGenerationPrompt: true})
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	if as("developer") != as("system") {
		t.Fatalf("developer rendered as %q, system as %q", as("developer"), as("system"))
	}
}

// Jinja's tojson sorts an object's keys and escapes four characters HTML cares
// about. Both are visible in the fixture; this pins them on their own so a
// failure says which of the two broke.
func TestToolJSONFollowsJinja(t *testing.T) {
	var b strings.Builder
	writeJSON(&b, map[string]any{
		"b": "it's <there> & back",
		"a": []any{true, nil, 2.0},
	})
	want := `{"a": [true, null, 2], "b": "it\u0027s \u003cthere\u003e \u0026 back"}`
	if got := b.String(); got != want {
		t.Fatalf("wrote %s\nwant   %s", got, want)
	}
}

func TestStripThinking(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"plain", "plain"},
		{"<think>\nhidden\n</think>\n\nafter", "after"},
		{"<think>\nunclosed", "<think>\nunclosed"},
		{"a</think>b</think>c", "c"},
	} {
		if got := StripThinking(c.in); got != c.want {
			t.Fatalf("StripThinking(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
