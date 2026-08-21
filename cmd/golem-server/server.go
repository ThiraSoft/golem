package main

// The HTTP layer. It knows the shapes of OpenAI's API and nothing about
// attention.
//
// One model and one cache mean one request at a time: a mutex serializes them
// and a second request waits. There is no queue and no batching between
// requests, which the README says plainly rather than implying otherwise.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/sample"
)

type Server struct {
	mu       sync.Mutex
	gen      *Generator
	vocab    Vocabulary
	name     string
	tpl      chat.Template
	defaults sample.Params
	served   int
}

func NewServer(g *Generator, v Vocabulary, name string, tpl chat.Template, defaults sample.Params) *Server {
	return &Server{gen: g, vocab: v, name: name, tpl: tpl, defaults: defaults}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("POST /v1/chat/completions", s.completions)
	return mux
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id":       s.name,
			"object":   "model",
			"created":  time.Now().Unix(),
			"owned_by": "golem",
		}},
	})
}

func (s *Server) completions(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error",
			"the request body is not the JSON this endpoint reads: "+err.Error())
		return
	}
	if err := check(&req); err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	prompt, err := s.tpl.Render(req.Messages, chat.Options{
		Tools:               req.Tools,
		AddGenerationPrompt: true,
	})
	if err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.served++
	id := "chatcmpl-" + strconv.Itoa(s.served)

	ids := s.vocab.Encode(prompt, false, true)
	params := s.sampling(&req)
	gen := s.gen
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		gen = gen.WithMaxTokens(*req.MaxTokens)
	}
	if req.Stream {
		s.stream(r.Context(), w, gen, id, ids, params, req.Stop)
		return
	}
	answer, err := gen.Generate(r.Context(), ids, params, req.Stop, nil)
	if err != nil {
		// A client that hung up gets no answer, and no error either: there is
		// nobody left to read one.
		if r.Context().Err() != nil {
			return
		}
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	note(w, answer)
	reason := answer.Reason
	writeJSON(w, http.StatusOK, completionResponse{
		ID: id, Object: "chat.completion", Created: time.Now().Unix(), Model: s.name,
		Choices: []choice{{
			Index: 0,
			Message: &responseMessage{Role: "assistant", Content: answer.Text,
				ToolCalls: answer.ToolCalls},
			FinishReason: &reason,
		}},
		Usage: &usage{PromptTokens: answer.Prompt, CompletionTokens: answer.Generated,
			TotalTokens: answer.Prompt + answer.Generated},
	})
}

// check refuses what would change the shape of the answer, rather than
// answering something the client did not ask for.
func check(req *completionRequest) error {
	if len(req.Messages) == 0 {
		return fmt.Errorf("a conversation with no message")
	}
	if req.N != nil && *req.N != 1 {
		return fmt.Errorf("n is %d: this server draws one answer", *req.N)
	}
	if req.LogProbs != nil && *req.LogProbs {
		return fmt.Errorf("logprobs is not implemented")
	}
	switch pick := req.ToolChoice.(type) {
	case nil:
	case string:
		if pick != "auto" && pick != "none" {
			return fmt.Errorf("tool_choice %q is not implemented; auto and none are", pick)
		}
		if pick == "none" {
			req.Tools = nil
		}
	default:
		return fmt.Errorf("tool_choice as an object is not implemented; auto and none are")
	}
	return nil
}

// sampling starts from what the file asks for and lets the request override.
func (s *Server) sampling(req *completionRequest) sample.Params {
	p := s.defaults
	if req.Temperature != nil {
		p.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		p.TopP = float32(*req.TopP)
	}
	if req.TopK != nil {
		p.TopK = *req.TopK
	}
	if req.Seed != nil {
		p.Seed = *req.Seed
	}
	return p
}

// note tells the log line what the answer cost. Reading a prompt and drawing
// an answer are the two halves of the wait, and they have nothing in common:
// only their separate rates say which one to go after.
func note(w http.ResponseWriter, a Answer) {
	rec, ok := w.(*recorder)
	if !ok {
		return
	}
	rate := func(n int, d time.Duration) float64 {
		if d <= 0 {
			return 0
		}
		return float64(n) / d.Seconds()
	}
	rec.note = fmt.Sprintf("%d prompt in %s (%.1f/s), %d drawn in %s (%.2f/s), %s",
		a.Prompt, a.Prefill.Round(time.Millisecond), rate(a.Prompt, a.Prefill),
		a.Generated, a.Decode.Round(time.Millisecond), rate(a.Generated, a.Decode),
		a.Reason)
}

func writeJSON(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(body)
}

func refuse(w http.ResponseWriter, code int, kind, message string) {
	// The log line for this request carries the reason too, so a refusal is
	// visible on the server and not only to the client.
	if rec, ok := w.(*recorder); ok {
		rec.reason = message
	}
	var body apiError
	body.Error.Message = message
	body.Error.Type = kind
	writeJSON(w, code, body)
}
