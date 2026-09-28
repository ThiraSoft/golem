package main

// POST /v1/systemone: typed questions about a state, answered by Laya.
//
// It is TypeSafe Jev's endpoint, and the one laya-serve answers on, with the
// same request and the same answer: {"state": …, "questions": {…}} in, every
// option of every question with its probability out. A client of either needs
// only its base URL changed. A refusal is written as both write theirs,
// {"detail": "…"}, and with their codes: 400 for a body that is not the
// request, 413 for one past the limits below, 422 for a question Laya cannot
// read.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/ThiraSoft/golem/laya"
)

// Decider answers typed questions. laya.Model is one; the tests have another.
type Decider interface {
	Decide(ctx context.Context, state json.RawMessage, questions []laya.Named) (*laya.Result, error)
}

// SetDecider lets this server answer /v1/systemone.
func (s *Server) SetDecider(d Decider) { s.decider = d }

// The limits are laya-serve's: a body of a mebibyte, sixty-four questions.
// Every question is a sequence of its own in the pass, so the count is what
// bounds the work one request can ask for.
const (
	maxDecisionBody      = 1 << 20
	maxDecisionQuestions = 64
)

func (s *Server) systemone(w http.ResponseWriter, r *http.Request) {
	var req struct {
		State     json.RawMessage `json:"state"`
		Questions json.RawMessage `json:"questions"`
	}
	body := http.MaxBytesReader(w, r.Body, maxDecisionBody)
	if err := json.NewDecoder(body).Decode(&req); err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			detail(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		detail(w, http.StatusBadRequest, "request body must be valid JSON")
		return
	}
	if len(req.Questions) == 0 {
		detail(w, http.StatusBadRequest, "request body must be an object with a 'questions' field")
		return
	}
	if len(req.State) == 0 {
		detail(w, http.StatusBadRequest, "'state' is required")
		return
	}
	questions, err := laya.ParseQuestions(req.Questions)
	if err != nil {
		detail(w, http.StatusBadRequest, "'questions' must be an object")
		return
	}
	if len(questions) > maxDecisionQuestions {
		detail(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("too many questions (%d > %d)", len(questions), maxDecisionQuestions))
		return
	}

	res, err := s.decider.Decide(r.Context(), req.State, questions)
	if err != nil {
		var bad *laya.RequestError
		switch {
		case errors.As(err, &bad):
			detail(w, http.StatusUnprocessableEntity, err.Error())
		case r.Context().Err() != nil:
			// The client left; nobody is reading an answer.
		default:
			detail(w, http.StatusInternalServerError, err.Error())
		}
		return
	}
	if rec, ok := w.(*recorder); ok {
		plural := "s"
		if len(questions) == 1 {
			plural = ""
		}
		rec.note = fmt.Sprintf("%d question%s, %d positions", len(questions), plural, res.InputTokens)
	}
	writeJSON(w, http.StatusOK, res)
}

// detail refuses a request the way FastAPI does, which is what a client of
// Jev or laya-serve reads a refusal as.
func detail(w http.ResponseWriter, code int, message string) {
	if rec, ok := w.(*recorder); ok {
		rec.reason = message
	}
	writeJSON(w, code, map[string]string{"detail": message})
}
