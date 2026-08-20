// Command serve answers an OpenAI-compatible API over a Gemma 4 GGUF, on the
// CPU.
//
//	serve -model gemma-4-E2B-it-QAT-Q4_0.gguf -addr :8080
//
// It implements /v1/chat/completions, streamed or not, tool declarations
// included, and /v1/models. It reports the calls the model makes; running them
// is the client's part, as the protocol has it.
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

	"github.com/ThiraSoft/golem/gemma"
	"github.com/ThiraSoft/golem/token/bpe"
)

func main() {
	model := flag.String("model", os.Getenv("GOLEM_MODEL"), "Gemma 4 GGUF file (or GOLEM_MODEL)")
	addr := flag.String("addr", "127.0.0.1:8080", "address to listen on")
	context := flag.Int("context", 4096, "positions to keep; the file declares 131072, which no machine here would survive")
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
	m, err := gemma.Open(*model, *context)
	if err != nil {
		fail(err)
	}
	defer m.Close()
	vocab, err := bpe.Load(m.File())
	if err != nil {
		fail(err)
	}

	// The largest sliding window is what a rewind of the cache has to respect.
	window := 0
	for _, b := range m.Cfg.Blocks {
		if b.Window && b.WindowSize > window {
			window = b.WindowSize
		}
	}
	params := m.Cfg.Sampling
	params.Seed = rand.Uint64()

	ctx := NewContext(m, window, *context, time.Now, *ttl)
	gen := NewGenerator(ctx, vocab, m.Cfg.Vocab, *maxTokens)
	name := strings.TrimSuffix(filepath.Base(*model), ".gguf")
	server := NewServer(gen, vocab, name, m.Cfg.EmptyThought, params)

	fmt.Fprintf(os.Stderr, "%s: %d blocks, %d positions, loaded in %s on %d cores\n",
		name, len(m.Cfg.Blocks), *context,
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
	fmt.Fprintln(os.Stderr, "serve:", err)
	os.Exit(1)
}
