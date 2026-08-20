package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLoggingWritesOneLinePerRequest(t *testing.T) {
	var out strings.Builder
	s := newTestServer([]string{"hi", "<turn|>"})
	h := logging(&out, s.Handler())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`)))
	if line := out.String(); !strings.Contains(line, "POST /v1/chat/completions 200") {
		t.Fatalf("logged %q", line)
	}
}

// A refusal says why on the server too, which is the whole reason the line
// exists: a client seeing nothing needs someone to have written down what
// happened.
func TestLoggingCarriesTheReasonOfARefusal(t *testing.T) {
	var out strings.Builder
	s := newTestServer(nil)
	h := logging(&out, s.Handler())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"n":4}`)))
	line := out.String()
	if !strings.Contains(line, "400") || !strings.Contains(line, "this server draws one answer") {
		t.Fatalf("logged %q", line)
	}
}

// The middleware must not swallow the flush: without it a streamed answer
// would sit in a buffer until the whole thing was drawn.
func TestLoggingKeepsTheStreamFlushable(t *testing.T) {
	var out strings.Builder
	s := newTestServer([]string{"one", "<turn|>"})
	h := logging(&out, s.Handler())

	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"stream":true}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("%d: %s", w.Code, w.Body)
	}
	if !strings.Contains(w.Body.String(), "data: ") {
		t.Fatalf("the stream did not survive the middleware: %q", w.Body)
	}
	if !strings.Contains(w.Body.String(), "[DONE]") {
		t.Fatalf("the stream never closed: %q", w.Body)
	}
}
