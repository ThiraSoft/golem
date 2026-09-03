package main

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/stt"
)

// TestTranscriptionsWithoutStream checks the plain form: multipart in, one JSON body out.
func TestTranscriptionsWithoutStream(t *testing.T) {
	s := newTestServerWithSTT(t)
	body, contentType := multipartClip(t, "clip.wav", map[string]string{"model": "stt"})
	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", body)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(out.Text) == "" {
		t.Fatal("empty transcript")
	}
}

// TestTranscriptionsStreaming checks the event sequence: deltas,
// then one done carrying the whole text, then the terminator.
func TestTranscriptionsStreaming(t *testing.T) {
	s := newTestServerWithSTT(t)
	body, contentType := multipartClip(t, "clip.wav", map[string]string{"model": "stt", "stream": "true"})
	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", body)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	events := parseSSE(t, rec.Body.String())
	var deltas int
	var done string
	for _, e := range events {
		switch e.Type {
		case "transcript.text.delta":
			deltas++
		case "transcript.text.done":
			done = e.Text
		}
	}
	if deltas == 0 {
		t.Error("no delta events")
	}
	if strings.TrimSpace(done) == "" {
		t.Error("no done event with the whole text")
	}
	if !strings.HasSuffix(strings.TrimSpace(rec.Body.String()), "data: [DONE]") {
		t.Error("stream does not end with [DONE]")
	}
}

// TestTranscriptionsWithoutModel: with no -stt, the route is not there.
func TestTranscriptionsWithoutModel(t *testing.T) {
	s := newTestServer(t, nil)
	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", strings.NewReader(""))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 404 {
		t.Fatalf("status %d, want 404", rec.Code)
	}
}

func newTestServerWithSTT(t *testing.T) *Server {
	t.Helper()
	dir := os.Getenv("GOLEM_STT")
	if dir == "" {
		t.Skip("GOLEM_STT not set")
	}
	o, err := stt.Locate(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := stt.Open(o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	s := newTestServer(t, nil)
	s.SetSTT(m)
	return s
}

func multipartClip(t *testing.T, name string, fields map[string]string) (io.Reader, string) {
	t.Helper()
	raw, err := os.ReadFile("../../testdata/stt/" + name)
	if err != nil {
		t.Skipf("no clip: %v", err)
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	part, err := w.CreateFormFile("file", name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(raw); err != nil {
		t.Fatal(err)
	}
	for k, v := range fields {
		if err := w.WriteField(k, v); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return &buf, w.FormDataContentType()
}

type sseEvent struct {
	Type  string `json:"type"`
	Text  string `json:"text"`
	Delta string `json:"delta"`
}

func parseSSE(t *testing.T, body string) []sseEvent {
	t.Helper()
	var out []sseEvent
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(strings.TrimSpace(line), "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var e sseEvent
		if err := json.Unmarshal([]byte(payload), &e); err != nil {
			t.Fatalf("event %q: %v", payload, err)
		}
		out = append(out, e)
	}
	return out
}
