package main

// What the server says about the requests it served.
//
// A server that prints nothing is a server nobody can debug from the outside:
// when a client sees no answer, the first question is whether the request ever
// arrived, and only the server can answer that. So every request leaves one
// line, and a refused one leaves the reason with it.

import (
	"fmt"
	"io"
	"net/http"
	"time"
)

// recorder keeps what was answered, so the line can say it.
type recorder struct {
	http.ResponseWriter
	status  int
	written int
	// reason is the message a refusal carried, kept for the log line.
	reason string
}

func (r *recorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *recorder) Write(p []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	n, err := r.ResponseWriter.Write(p)
	r.written += n
	return n, err
}

// Flush passes the flush through, which server-sent events depend on.
func (r *recorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// logging writes one line per request: what was asked, what came back, and how
// long it took.
func logging(out io.Writer, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &recorder{ResponseWriter: w}
		next.ServeHTTP(rec, r)
		line := fmt.Sprintf("%s %s %d %d bytes in %s", r.Method, r.URL.Path,
			rec.status, rec.written, time.Since(start).Round(time.Millisecond))
		if rec.reason != "" {
			line += ": " + rec.reason
		}
		fmt.Fprintln(out, line)
	})
}
