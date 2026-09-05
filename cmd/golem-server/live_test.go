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
	ctx := NewContext(running(t, m.Forward), m.Window, 2048, time.Now, 0)
	server := NewServer(poolOf(NewGenerator(ctx, m.Vocab, m.Template, m.Vocabulary, 128)),
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

func TestLiveParallelSlotsVulkan(t *testing.T) {
	for _, key := range []string{"GOLEM_MODEL", "GOLEM_MODEL_QWEN"} {
		t.Run(key, func(t *testing.T) {
			path := os.Getenv(key)
			if path == "" {
				t.Skipf("%s is not set", key)
			}
			m, err := engine.Open(path, 4096, 2)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if err := m.UseVulkan(); err != nil {
				t.Skipf("no Vulkan: %v", err)
			}

			params := m.Sampling
			params.Temperature = 0

			calls := 0
			runner := NewRunner(m.Forward)
			stop := make(chan struct{})
			defer close(stop)
			go runner.Run(stop)

			slots := make([]*slot, 2)
			for i := range slots {
				ctx := NewSlotContext(runner, i, m.Window, m.SlotContext(), time.Now, 0)
				gen := NewGenerator(ctx, m.Vocab, m.Template, m.Vocabulary, 32)
				gen.calls = &calls
				slots[i] = &slot{index: i, ctx: ctx, gen: gen}
			}
			pool := NewPool(slots, time.Now)
			server := NewServer(pool, m.Vocab, "live", m.Template, params)

			body1 := `{"messages":[{"role":"user","content":"Count to 3: 1, 2,"}],"temperature":0,"max_tokens":10}`
			body2 := `{"messages":[{"role":"user","content":"Say Hello in French"}],"temperature":0,"max_tokens":10}`

			done := make(chan struct{}, 2)
			for _, body := range []string{body1, body2} {
				go func(b string) {
					req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(b))
					w := httptest.NewRecorder()
					server.Handler().ServeHTTP(w, req)
					if w.Code != 200 {
						t.Errorf("%d: %s", w.Code, w.Body)
					}
					done <- struct{}{}
				}(body)
			}
			<-done
			<-done
		})
	}
}
