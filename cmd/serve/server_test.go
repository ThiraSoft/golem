package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newTestServer(script []string) *Server {
	g, v := newGenerator(script, 64)
	return NewServer(g, v, "test-model", false, greedy())
}

func post(t *testing.T, s *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)
	return w
}

func TestModelsListsTheOneModel(t *testing.T) {
	s := newTestServer(nil)
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/v1/models", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var out struct {
		Object string `json:"object"`
		Data   []struct {
			ID     string `json:"id"`
			Object string `json:"object"`
		} `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "list" || len(out.Data) != 1 || out.Data[0].ID != "test-model" {
		t.Fatalf("%s", w.Body)
	}
}

func TestCompletionAnswers(t *testing.T) {
	s := newTestServer([]string{"hello", "<turn|>"})
	w := post(t, s, `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var out struct {
		Object  string `json:"object"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "chat.completion" {
		t.Fatalf("object %q", out.Object)
	}
	if len(out.Choices) != 1 || out.Choices[0].Message.Content != "hello" {
		t.Fatalf("%s", w.Body)
	}
	if out.Choices[0].Message.Role != "assistant" || out.Choices[0].FinishReason != "stop" {
		t.Fatalf("%s", w.Body)
	}
	if out.Usage.TotalTokens != out.Usage.PromptTokens+out.Usage.CompletionTokens {
		t.Fatalf("usage %+v", out.Usage)
	}
}

// A second request repeating the conversation pays for what it added, not for
// what it repeated. The measure is a fresh server given the same second
// request: it has no prefix to reuse and feeds the whole thing.
func TestASecondRequestReusesTheCache(t *testing.T) {
	const first = `{"messages":[{"role":"user","content":"a"}]}`
	const second = `{"messages":[{"role":"user","content":"a"},
		{"role":"assistant","content":"one"},{"role":"user","content":"b"}]}`
	script := []string{"one", "<turn|>", "two", "<turn|>"}

	warm := newTestServer(script)
	post(t, warm, first)
	reused := promptTokens(t, post(t, warm, second))

	cold := newTestServer(script)
	whole := promptTokens(t, post(t, cold, second))

	if reused >= whole {
		t.Fatalf("the warm server fed %d positions and a cold one %d: the prefix was not reused",
			reused, whole)
	}
}

func promptTokens(t *testing.T, w *httptest.ResponseRecorder) int {
	t.Helper()
	var out struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatalf("%s: %v", w.Body, err)
	}
	return out.Usage.PromptTokens
}

func TestCompletionReportsAToolCall(t *testing.T) {
	s := newTestServer([]string{`<|tool_call>call:weather{city:<|"|>Lyon<|"|>}<tool_call|>`, "<turn|>"})
	w := post(t, s, `{"model":"test-model","messages":[{"role":"user","content":"weather?"}],
		"tools":[{"type":"function","function":{"name":"weather","description":"w",
		"parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	c := out.Choices[0]
	if c.FinishReason != "tool_calls" || len(c.Message.ToolCalls) != 1 {
		t.Fatalf("%s", w.Body)
	}
	call := c.Message.ToolCalls[0]
	if call.Type != "function" || call.Function.Name != "weather" {
		t.Fatalf("%s", w.Body)
	}
	if call.Function.Arguments != `{"city":"Lyon"}` {
		t.Fatalf("arguments %q", call.Function.Arguments)
	}
	if call.ID == "" {
		t.Fatal("a call with no identifier cannot be answered")
	}
}

func TestCompletionRefusesWhatItDoesNotImplement(t *testing.T) {
	s := newTestServer([]string{"x", "<turn|>"})
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi"}],"n":2}`,
		`{"messages":[{"role":"user","content":"hi"}],"logprobs":true}`,
		`{"messages":[{"role":"user","content":"hi"}],"tool_choice":"required"}`,
		`{"messages":[]}`,
		`{"messages":[{"role":"tool","content":"x"}]}`,
		`{`,
	} {
		w := post(t, s, body)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("%s answered %d, want 400", body, w.Code)
		}
		var out struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
			} `json:"error"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil || out.Error.Message == "" {
			t.Fatalf("the error envelope was %s", w.Body)
		}
	}
}

func TestAnUnknownPathIs404(t *testing.T) {
	s := newTestServer(nil)
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/v1/embeddings", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("%d", w.Code)
	}
}
