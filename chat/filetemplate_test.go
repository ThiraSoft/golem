package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// The jinja fixture holds what Jinja2 made of each template from messages in
// transformers' shape. Replayed here as Messages, through FileTemplate, they
// have to come out the same: that is what says the conversion hands a
// template what transformers would.
type wireMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []ToolCall      `json:"tool_calls"`
	Name       string          `json:"name"`
	ToolCallID string          `json:"tool_call_id"`
	Reasoning  *string         `json:"reasoning_content"`
}

type wireCase struct {
	Name string `json:"name"`
	Vars struct {
		Messages            []wireMessage `json:"messages"`
		Tools               []Tool        `json:"tools"`
		EnableThinking      bool          `json:"enable_thinking"`
		AddGenerationPrompt bool          `json:"add_generation_prompt"`
		BOS                 string        `json:"bos_token"`
		EOS                 string        `json:"eos_token"`
	} `json:"vars"`
	Rendered *string `json:"rendered"`
}

// message reads one wire message as a Message, or says it cannot be one: a
// null content, a reasoning field, or parts in an order Message does not keep.
func (w wireMessage) message() (Message, bool) {
	m := Message{Role: w.Role, ToolCalls: w.ToolCalls, Name: w.Name, ToolCallID: w.ToolCallID}
	if w.Reasoning != nil || string(w.Content) == "null" {
		return m, false
	}
	if err := json.Unmarshal(w.Content, &m.Content); err == nil {
		return m, true
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(w.Content, &parts); err != nil {
		return m, false
	}
	stage := 0 // pictures, then recordings, then one text
	for _, p := range parts {
		switch {
		case p.Type == "image" && stage == 0:
			m.Images = append(m.Images, nil)
		case p.Type == "audio" && stage <= 1:
			stage = 1
			m.Audio = append(m.Audio, nil)
		case p.Type == "text" && stage <= 1 && p.Text != "":
			stage = 2
			m.Content = p.Text
		default:
			return m, false
		}
	}
	return m, true
}

type noCalls struct{}

func (noCalls) Render([]Message, Options) (string, error)          { return "", nil }
func (noCalls) ParseCalls(text string) (string, []ToolCall, error) { return text, nil, nil }
func (noCalls) CallOpen() string                                   { return "" }
func (noCalls) ReasoningMarkers() (string, string)                 { return "", "" }

func TestFileTemplateHandsTheTemplateWhatTransformersWould(t *testing.T) {
	dir := filepath.Join("..", "testdata", "jinja")
	raw, err := os.ReadFile(filepath.Join(dir, "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Templates []struct {
			File  string     `json:"file"`
			Cases []wireCase `json:"cases"`
		} `json:"templates"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	replayed := 0
	for _, file := range fixture.Templates {
		src, err := os.ReadFile(filepath.Join(dir, "templates", file.File))
		if err != nil {
			t.Fatal(err)
		}
		for i, c := range file.Cases {
			var msgs []Message
			ok := true
			for _, w := range c.Vars.Messages {
				m, fits := w.message()
				ok = ok && fits
				msgs = append(msgs, m)
			}
			if !ok || c.Rendered == nil {
				continue
			}
			t.Run(file.File+"/"+c.Name+"/"+strconv.Itoa(i%4), func(t *testing.T) {
				tokens := map[string]string{"bos_token": c.Vars.BOS, "eos_token": c.Vars.EOS}
				tpl, err := NewFileTemplate(string(src), noCalls{}, tokens, nil)
				if err != nil {
					t.Fatal(err)
				}
				got, err := tpl.Render(msgs, Options{
					EnableThinking:      c.Vars.EnableThinking,
					Tools:               c.Vars.Tools,
					AddGenerationPrompt: c.Vars.AddGenerationPrompt,
				})
				if err != nil {
					t.Fatal(err)
				}
				if got != *c.Rendered {
					t.Fatalf("rendered\n%q\nJinja2 rendered\n%q", got, *c.Rendered)
				}
			})
			replayed++
		}
	}
	if replayed < 400 {
		t.Fatalf("only %d cases could be replayed as Messages; the fixture has drifted from what Message carries", replayed)
	}
}

func TestFileTemplateSwapsTheMediaPlaceholder(t *testing.T) {
	src := "{% for m in messages %}{% for p in m.content %}{% if p.type == 'image' %}<|image|>{% else %}{{ p.text }}{% endif %}{% endfor %}{% endfor %}"
	tpl, err := NewFileTemplate(src, noCalls{}, nil, strings.NewReplacer("<|image|>", "<|image><image|>"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := tpl.Render([]Message{{Role: "user", Content: "hi", Images: [][]byte{{1}}}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if got != "<|image><image|>hi" {
		t.Fatalf("got %q", got)
	}
}
