// Command golem-server answers an OpenAI-compatible API over a GGUF, on the
// CPU.
//
//	golem-server -model gemma-4-E2B-it-QAT-Q4_0.gguf -addr :8080
//	golem-server -model Qwen3-4B-Q4_0.gguf -addr :8080
//
// Which engine reads the file is read from the file: it declares its own
// architecture, and gemma4 and qwen3 are the two that are implemented. It
// serves /v1/chat/completions, streamed or not, tool declarations included,
// and /v1/models. It reports the calls the model makes; running them is the
// client's part, as the protocol has it.
package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/engine"
)

func main() {
	model := flag.String("model", os.Getenv("GOLEM_MODEL"), "GGUF file, gemma4 or qwen3 (or GOLEM_MODEL)")
	addr := flag.String("addr", "127.0.0.1:8080", "address to listen on")
	context := flag.Int("context", 4096, "positions to keep; the files declare far more than any machine here would survive")
	maxTokens := flag.Int("n", 1024, "most tokens to draw for one answer, when the request names no limit")
	ttl := flag.Duration("cache-ttl", 0, "forget a conversation's tokens after this long idle; 0 never forgets. The memory is allocated at startup and is released by neither")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s [options]\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()

	if *model == "" {
		fail(fmt.Errorf("no model: pass -model, or set GOLEM_MODEL"))
	}
	start := time.Now()
	m, err := engine.Open(*model, *context)
	if err != nil {
		fail(err)
	}
	defer m.Close()

	params := m.Sampling
	params.Seed = rand.Uint64()

	ctx := NewContext(m.Forward, m.Window, *context, time.Now, *ttl)
	gen := NewGenerator(ctx, m.Vocab, m.Template, m.Vocabulary, *maxTokens)
	name := strings.TrimSuffix(filepath.Base(*model), ".gguf")
	server := NewServer(gen, m.Vocab, name, m.Template, params)

	fmt.Fprintf(os.Stderr, "%s: %s, %d blocks, %d positions, loaded in %s on %d cores\n",
		name, m.Name, m.Blocks, *context,
		time.Since(start).Round(time.Millisecond), runtime.NumCPU())
	// The port is taken before it is announced: an address already in use must
	// not be reported as a server that is listening.
	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		fail(err)
	}
	fmt.Fprintf(os.Stderr, "listening on http://%s/v1 — one request at a time\n", listener.Addr())
	if err := http.Serve(listener, logging(os.Stderr, server.Handler())); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "golem-server:", err)
	os.Exit(1)
}
