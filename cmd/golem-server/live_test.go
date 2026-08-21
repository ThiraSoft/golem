package main

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/engine"
)

// The tests with weights: a conversation declaring a tool, whose answer is a
// call this server can read. One per architecture, because a template that
// renders is not yet a template a model answers, and each skips when its file
// is not on the machine.
func TestLiveToolCall(t *testing.T) {
	for _, key := range []string{"GOLEM_MODEL", "GOLEM_MODEL_QWEN"} {
		t.Run(key, func(t *testing.T) {
			path := os.Getenv(key)
			if path == "" {
				t.Skipf("%s is not set", key)
			}
			liveToolCall(t, path)
		})
	}
}

func liveToolCall(t *testing.T, path string) {
	t.Helper()
	m, err := engine.Open(path, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	params := m.Sampling
	params.Temperature = 0
	ctx := NewContext(m.Forward, m.Window, 2048, time.Now, 0)
	server := NewServer(NewGenerator(ctx, m.Vocab, m.Template, m.Vocabulary, 128),
		m.Vocab, "live", m.Template, params)

	body := `{"messages":[{"role":"user","content":"What is the weather in Lyon right now?"}],
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
				Content   string          `json:"content"`
				ToolCalls []chat.ToolCall `json:"tool_calls"`
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
