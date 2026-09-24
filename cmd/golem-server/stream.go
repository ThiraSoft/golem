package main

// Server-sent events.
//
// Prose leaves as it is drawn. A call leaves once, whole, in a single delta:
// OpenAI's API allows a call to arrive in fragments and clients do reassemble
// them, but a fragment of a function's arguments is a thing nothing can check,
// and there is nothing to gain by sending one.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/ThiraSoft/golem/engine"
	"github.com/ThiraSoft/golem/sample"
)

func (s *Server) stream(ctx context.Context, w http.ResponseWriter, gen *Generator, id string, prompt engine.Prompt, p sample.Params, stop []string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		refuse(w, http.StatusInternalServerError, "server_error", "this connection cannot be streamed to")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	created := time.Now().Unix()
	send := func(c choice) error {
		body, err := json.Marshal(completionResponse{
			ID: id, Object: "chat.completion.chunk", Created: created,
			Model: s.name, Choices: []choice{c},
		})
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", body); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}
	done := func() {
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}

	if err := send(choice{Delta: &responseMessage{Role: "assistant"}}); err != nil {
		return
	}
	answer, err := gen.GeneratePrompt(ctx, prompt, p, stop, func(d Delta) error {
		return send(choice{Delta: &responseMessage{Content: d.Content, ReasoningContent: d.Reasoning}})
	})
	if err != nil {
		if ctx.Err() != nil {
			return // the client left; there is no stream to write to
		}
		// The status line is already out, so the error goes down the stream,
		// which is the only place left for it. It goes as its own event, the
		// way llama.cpp and OpenAI send one, and not in the prose, where a
		// client would show it or read it aloud as if the model had said it.
		// Like llama.cpp, nothing follows it, not even [DONE].
		if rec, ok := w.(*recorder); ok {
			rec.reason = err.Error()
		}
		var body apiError
		body.Error.Message = err.Error()
		body.Error.Type = "server_error"
		if raw, err := json.Marshal(body); err == nil {
			fmt.Fprintf(w, "data: %s\n\n", raw)
			flusher.Flush()
		}
		return
	}
	note(w, answer)
	if len(answer.ToolCalls) > 0 {
		for i := range answer.ToolCalls {
			answer.ToolCalls[i].Index, answer.ToolCalls[i].HasIndex = i, true
		}
		if err := send(choice{Delta: &responseMessage{ToolCalls: answer.ToolCalls}}); err != nil {
			return
		}
	}
	reason := answer.Reason
	send(choice{Delta: &responseMessage{}, FinishReason: &reason})
	done()
}
