package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/sample"
)

func newTestServer(t testing.TB, script []string, words ...string) *Server {
	g, v := newGenerator(t, script, 64, words...)
	return NewServer(poolOf(g), v, "test-model", wordTemplate{}, greedy())
}

// newTestServerWithSlots is the same over several conversations at once.
func newTestServerWithSlots(t testing.TB, script []string, slots int) *Server {
	g, v := newGenerator(t, script, 64)
	pool := NewPool(slotsOf(g, slots), time.Now)
	return NewServer(pool, v, "test-model", wordTemplate{}, greedy())
}

// slotsOf is n slots over the same engine, each with its own conversation.
func slotsOf(g *Generator, n int) []*slot {
	out := make([]*slot, n)
	for i := range out {
		// A generator of its own, as main.go builds one per slot: the buffer
		// it scores into is not something two conversations may share.
		ctx := NewSlotContext(g.ctx.runner, i, 0, 4096, time.Now, 0)
		gen := NewGenerator(ctx, g.vocab, g.tpl, len(g.logits), g.maxTokens)
		gen.calls = g.calls
		out[i] = &slot{index: i, ctx: ctx, gen: gen}
	}
	return out
}

// poolOf is the one-slot pool a test that does not care about slots wants.
func poolOf(g *Generator) *Pool {
	return NewPool([]*slot{{index: 0, ctx: g.ctx, gen: g}}, time.Now)
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
	s := newTestServer(t, nil)
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
	s := newTestServer(t, []string{"hello", "<turn|>"})
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

	warm := newTestServer(t, script)
	post(t, warm, first)
	reused := promptTokens(t, post(t, warm, second))

	cold := newTestServer(t, script)
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
	s := newTestServer(t, []string{"CALLweather{city=Lyon}", "<turn|>"})
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
	s := newTestServer(t, []string{"x", "<turn|>"})
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
	s := newTestServer(t, nil)
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/v1/embeddings", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("%d", w.Code)
	}
}

// The four penalty fields a request may carry, over the file's own values.
func TestSamplingReadsThePenaltyFields(t *testing.T) {
	s := newTestServer(t, []string{"a", "<turn|>"})
	s.defaults = sample.Params{Temperature: 1, PenaltyLastN: 64, PenaltyRepeat: 1}

	lastN, repeat, freq, present := 128, 1.2, 0.3, 0.4
	got := s.sampling(&completionRequest{RepeatLastN: &lastN, RepeatPenalty: &repeat,
		FrequencyPenalty: &freq, PresencePenalty: &present})

	if got.PenaltyLastN != 128 || got.PenaltyRepeat != 1.2 ||
		got.PenaltyFreq != 0.3 || got.PenaltyPresent != 0.4 {
		t.Fatalf("the request asked for 128/1.2/0.3/0.4 and the sampler took %+v", got)
	}
	if none := s.sampling(&completionRequest{}); none.PenaltyRepeat != 1 || none.PenaltyLastN != 64 {
		t.Fatalf("a request naming nothing changed the file's values: %+v", none)
	}
}

// answerOf reads the assistant's text out of a completion.
func answerOf(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil || len(out.Choices) == 0 {
		t.Fatalf("the answer was %s", w.Body)
	}
	return out.Choices[0].Message.Content
}

// The model is scripted to say no; the grammar allows only yes. What comes back
// is what the grammar allows, which is the whole point of the thing.
func TestAGrammarDecidesTheAnswer(t *testing.T) {
	s := newTestServer(t, []string{"no", "no", "<turn|>"}, "yes")
	w := post(t, s, `{"model":"test-model","messages":[{"role":"user","content":"hi"}],
		"grammar":"root ::= \"yes\""}`)
	if got := answerOf(t, w); got != "yes" {
		t.Fatalf("the answer is %q, and the grammar allows only yes", got)
	}
}

// A schema arrives as JSON and leaves as a grammar; here it allows one word and
// the model wanted another.
func TestAJSONSchemaConstrainsTheAnswer(t *testing.T) {
	s := newTestServer(t, []string{"no", "no", "<turn|>"}, "42")
	w := post(t, s, `{"model":"test-model","messages":[{"role":"user","content":"hi"}],
		"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"const":42}}}}`)
	if got := answerOf(t, w); !strings.Contains(got, "42") {
		t.Fatalf("the answer is %q, and the schema names the constant 42", got)
	}
}

// What is refused, and refused before a slot is taken: a request that will not
// be answered has no business waiting behind one that will.
func TestWhatAConstrainedRequestIsRefusedFor(t *testing.T) {
	s := newTestServer(t, []string{"no", "<turn|>"})
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi"}],"grammar":"root ::= ("}`,
		`{"messages":[{"role":"user","content":"hi"}],"grammar":"root ::= \"a\"","response_format":{"type":"json_object"}}`,
		`{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema"}}`,
		`{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"string","pattern":"^a$"}}}}`,
		`{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"yaml"}}`,
	} {
		if w := post(t, s, body); w.Code != http.StatusBadRequest {
			t.Fatalf("%s answered %d, want 400", body, w.Code)
		}
	}
}

// A request naming no format is not constrained at all, and builds no table.
func TestAnUnconstrainedRequestBuildsNoTokenTable(t *testing.T) {
	s := newTestServer(t, []string{"hello", "<turn|>"})
	post(t, s, `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`)
	if s.tokens != nil {
		t.Fatal("a request that asked for no grammar built the vocabulary table anyway")
	}
}
