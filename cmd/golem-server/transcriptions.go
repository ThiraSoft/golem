package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ThiraSoft/golem/stt"
)

// transcribe and transcribeStream take the group when the server was given one,
// and the model alone when it was not.
func (s *Server) transcribe(raw []byte) (string, error) {
	if s.sttGroup == nil {
		return s.stt.Transcribe(raw)
	}
	return s.sttGroup.Transcribe(raw)
}

func (s *Server) transcribeStream(ctx context.Context, raw []byte, each func(stt.Segment)) error {
	if s.sttGroup == nil {
		return s.stt.TranscribeStream(ctx, raw, each)
	}
	return s.sttGroup.TranscribeStream(ctx, raw, each)
}

// POST /v1/audio/transcriptions — OpenAI-compatible endpoint.
// Reads multipart form with fields: file, model, response_format, stream.
func (s *Server) transcriptions(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	file, _, err := r.FormFile("file")
	if err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", "no file part")
		return
	}
	defer file.Close()
	raw, err := io.ReadAll(file)
	if err != nil {
		refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if r.FormValue("stream") != "true" {
		text, err := s.transcribe(raw)
		if err != nil {
			if errors.Is(err, stt.ErrGroupFull) {
				refuse(w, http.StatusTooManyRequests, "rate_limit_error", err.Error())
				return
			}
			refuse(w, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"text": text})
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)
	send := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if flusher != nil {
			flusher.Flush()
		}
	}

	var whole strings.Builder
	err = s.transcribeStream(r.Context(), raw, func(seg stt.Segment) {
		whole.WriteString(seg.Text)
		send(map[string]any{"type": "transcript.text.delta", "delta": seg.Text})
	})
	if err != nil {
		send(map[string]any{"type": "error", "error": err.Error()})
	} else {
		send(map[string]any{"type": "transcript.text.done", "text": strings.TrimSpace(whole.String())})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}
