package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/laya"
	"github.com/ThiraSoft/golem/sample"
)

type fakeDecider struct {
	state json.RawMessage
	got   []laya.Named
	err   error
}

func (f *fakeDecider) Decide(_ context.Context, state json.RawMessage, qs []laya.Named) (*laya.Result, error) {
	f.state, f.got = state, qs
	if f.err != nil {
		return nil, f.err
	}
	res := &laya.Result{InputTokens: 42}
	for _, q := range qs {
		res.Answers = append(res.Answers, laya.Answer{ID: q.ID, Type: "noul", Noul: 0.25})
	}
	return res, nil
}

func decide(t *testing.T, f *fakeDecider, body string) *httptest.ResponseRecorder {
	t.Helper()
	s := NewServer(nil, nil, "laya", nil, sample.Params{})
	s.SetDecider(f)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(body)))
	return rec
}

func TestSystemOneAnswersInJevsShape(t *testing.T) {
	f := &fakeDecider{}
	rec := decide(t, f, `{"state":{"body":"charged twice"},"questions":{
		"b":{"type":"noul","instructions":"About billing."},
		"a":{"type":"noul","instructions":"About sales."}},"unknown":1}`)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if string(f.state) != `{"body":"charged twice"}` {
		t.Fatalf("the state reached the model as %s", f.state)
	}
	// The questions keep the order they were written in, not the keys' order.
	if len(f.got) != 2 || f.got[0].ID != "b" || f.got[1].ID != "a" {
		t.Fatalf("the questions reached the model as %+v", f.got)
	}
	var out struct {
		Answers map[string]struct {
			Type string  `json:"type"`
			Noul float64 `json:"noul"`
		} `json:"answers"`
		Usage struct {
			InputTokens int `json:"input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Answers["b"].Noul != 0.25 || out.Answers["a"].Type != "noul" || out.Usage.InputTokens != 42 {
		t.Fatalf("answered %s", rec.Body)
	}
}

func TestSystemOneRefusesAsLayaServeDoes(t *testing.T) {
	many := make([]string, maxDecisionQuestions+1)
	for i := range many {
		many[i] = fmt.Sprintf(`"q%d":{"type":"noul","instructions":"x"}`, i)
	}
	for _, c := range []struct {
		name, body string
		err        error
		code       int
	}{
		{"not json", `{`, nil, 400},
		{"no questions", `{"state":"x"}`, nil, 400},
		{"no state", `{"questions":{"q":{"type":"noul","instructions":"x"}}}`, nil, 400},
		{"questions not an object", `{"state":"x","questions":[1]}`, nil, 400},
		{"too many questions", `{"state":"x","questions":{` + strings.Join(many, ",") + `}}`, nil, 413},
		{"too large", `{"state":"` + strings.Repeat("x", maxDecisionBody) + `","questions":{}}`, nil, 413},
		{"a question laya cannot read", `{"state":"x","questions":{"q":{"type":"rank","instructions":"x"}}}`,
			&laya.RequestError{Err: fmt.Errorf(`laya: question "q": type "rank" is none of choice, score and noul`)}, 422},
		{"the model failing", `{"state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}`,
			fmt.Errorf("vulkan: device lost"), 500},
	} {
		t.Run(c.name, func(t *testing.T) {
			rec := decide(t, &fakeDecider{err: c.err}, c.body)
			if rec.Code != c.code {
				t.Fatalf("status %d, want %d: %s", rec.Code, c.code, rec.Body)
			}
			var out struct {
				Detail string `json:"detail"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil || out.Detail == "" {
				t.Fatalf("the refusal carries no detail: %s", rec.Body)
			}
		})
	}
}

func TestSystemOneIsNotServedWithoutLaya(t *testing.T) {
	s := NewServer(nil, nil, "x", nil, sample.Params{})
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(`{}`)))
	if rec.Code != 404 {
		t.Fatalf("status %d without a decider", rec.Code)
	}
}
