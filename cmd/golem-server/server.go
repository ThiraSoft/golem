package main

// The HTTP layer. It knows the shapes of OpenAI's API and nothing about
// attention.
//
// One model means one request at a time: a slot is held for the whole of an
// answer and a second request waits for one. What several slots buy is the
// cache, not the throughput — slots.go says how one is chosen, and there is
// still no batching between requests.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/ThiraSoft/golem/chat"
	"github.com/ThiraSoft/golem/engine"
	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/sample"
)

type Server struct {
	mu       sync.Mutex // the identifiers, not the model: the pool serializes that
	pool     *Pool
	vocab    Vocabulary
	name     string
	tpl      chat.Template
	defaults sample.Params
	served   int
	// vision is nil unless a projector was opened, and images is what the
	// tower has already made of the pictures it was sent.
	vision engine.Media
	images *imageCache
}

func NewServer(pool *Pool, v Vocabulary, name string, tpl chat.Template, defaults sample.Params) *Server {
	return &Server{pool: pool, vocab: v, name: name, tpl: tpl, defaults: defaults,
		images: newImageCache(32)}
}

// SetVision lets this server answer about pictures.
func (s *Server) SetVision(v engine.Media) { s.vision = v }

// look runs the tower over every picture the conversation carries, in the
// order the turns carry them, which is the order the markers appear in the
// rendered prompt.
func (s *Server) look(msgs []chat.Message) ([][][]float32, error) {
	var out [][][]float32
	for _, m := range msgs {
		for _, raw := range m.Images {
			rows, err := s.images.Encode(raw, s.vision.EncodeImage)
			if err != nil {
				return nil, err
			}
			out = append(out, rows)
		}
	}
	return out, nil
}

// listen runs the encoder over every recording the conversation carries, in
// the order the turns carry them.
func (s *Server) listen(msgs []chat.Message) ([][][]float32, error) {
	var out [][][]float32
	for _, m := range msgs {
		for _, raw := range m.Audio {
			rows, err := s.vision.EncodeAudio(raw)
			if err != nil {
				return nil, err
			}
			out = append(out, rows)
		}
	}
	return out, nil
}

// carriesImages reports whether any turn brought a picture.
func carriesImages(msgs []chat.Message) bool {
	for _, m := range msgs {
		if len(m.Images) > 0 {
			return true
		}
	}
	return false
}

// carriesAudio reports whether any turn brought a recording.
func carriesAudio(msgs []chat.Message) bool {
	for _, m := range msgs {
		if len(m.Audio) > 0 {
			return true
		}
	}
	return false
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

	ids := s.vocab.Encode(prompt, false, true)

	// The rendered conversation carries an empty pair of markers where each
	// picture and each recording goes, and the rows go between them.
	built := &gemma.Prompt{Tokens: ids}
	if seen, heard := carriesImages(req.Messages), carriesAudio(req.Messages); seen || heard {
		if seen && (s.vision == nil || !s.vision.CanSee()) {
			refuse(w, http.StatusBadRequest, "invalid_request_error",
				"this model was started without a projector that can see, so it cannot look at a picture: start the server with -mmproj")
			return
		}
		if heard && (s.vision == nil || !s.vision.CanHear()) {
			refuse(w, http.StatusBadRequest, "invalid_request_error",
				"this model was started without a projector that can hear, so it cannot listen: start the server with -mmproj")
			return
		}
		var pictures, recordings [][][]float32
		if seen {
			pictures, err = s.look(req.Messages)
			if err != nil {
				refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
				return
			}
		}
		if heard {
			recordings, err = s.listen(req.Messages)
			if err != nil {
				refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
				return
			}
		}
		built, err = s.vision.Prompt(ids, pictures, recordings)
		if err != nil {
			refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		ids = built.Tokens
	}
	slot, waited, err := s.pool.Acquire(r.Context(), ids)
	if err != nil {
		// The client hung up while waiting for a slot. Nobody is left to read
		// an answer, or an error either.
		return
	}
	defer s.pool.Release(slot)
	waitedFor(w, waited, slot.index)

	// The runner counts the conversations in flight, so that a pass built for
	// one of them waits for the others rather than going alone.
	slot.ctx.runner.Enter()
	defer slot.ctx.runner.Leave()

	s.mu.Lock()
	s.served++
	id := "chatcmpl-" + strconv.Itoa(s.served)
	s.mu.Unlock()

	params := s.sampling(&req)
	gen := slot.gen
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		gen = gen.WithMaxTokens(*req.MaxTokens)
	}
	if req.Stream {
		s.stream(r.Context(), w, gen, id, built, params, req.Stop)
		return
	}
	answer, err := gen.GeneratePrompt(r.Context(), built, params, req.Stop, nil)
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

// waitedFor tells the log line which slot answered, and how long the request
// waited for it. A wait too short to round to a millisecond is not a queue and
// is left unsaid.
func waitedFor(w http.ResponseWriter, waited time.Duration, index int) {
	rec, ok := w.(*recorder)
	if !ok {
		return
	}
	rec.slot = fmt.Sprintf("slot %d", index)
	if waited.Round(time.Millisecond) > 0 {
		rec.slot += fmt.Sprintf(", queued %s", waited.Round(time.Millisecond))
	}
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
