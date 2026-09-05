package main

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/engine"
)

// The test that says whether any of this works: a real checkpoint, a real
// schema, and an answer encoding/json reads into a struct with the right shape.
// A model asked for JSON in prose usually obliges; a model asked for it through
// a grammar cannot do otherwise, and that is what is being checked.
func TestLiveJSONSchema(t *testing.T) {
	for _, key := range []string{"GOLEM_MODEL", "GOLEM_MODEL_QWEN"} {
		t.Run(key, func(t *testing.T) {
			path := os.Getenv(key)
			if path == "" {
				t.Skipf("%s is not set", key)
			}
			liveJSONSchema(t, path)
		})
	}
}

func liveJSONSchema(t *testing.T, path string) {
	t.Helper()
	m, err := engine.Open(path, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	params := m.Sampling
	params.Temperature = 0
	ctx := NewContext(running(t, m.Forward), m.Window, 2048, time.Now, 0)
	server := NewServer(poolOf(NewGenerator(ctx, m.Vocab, m.Template, m.Vocabulary, 256)),
		m.Vocab, "live", m.Template, params)

	body := `{"messages":[{"role":"user","content":"Lyon has about 520000 people and it rained today. Describe it."}],
		"temperature":0,
		"response_format":{"type":"json_schema","json_schema":{"name":"city","schema":{
			"type":"object",
			"properties":{
				"city":{"type":"string"},
				"population":{"type":"integer"},
				"raining":{"type":"boolean"}},
			"required":["city","population","raining"],
			"additionalProperties":false}}}}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.Handler().ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	answer := out.Choices[0].Message.Content
	t.Logf("answer: %s", answer)

	var city struct {
		City       *string `json:"city"`
		Population *int    `json:"population"`
		Raining    *bool   `json:"raining"`
	}
	if err := json.Unmarshal([]byte(answer), &city); err != nil {
		t.Fatalf("the answer is not JSON: %v\n%s", err, answer)
	}
	if city.City == nil || city.Population == nil || city.Raining == nil {
		t.Fatalf("the schema requires three properties and the answer holds %s", answer)
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Fatalf("the answer ended with reason %q, so the grammar did not close it",
			out.Choices[0].FinishReason)
	}
}
