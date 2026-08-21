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
	parallel := flag.Int("parallel", 1, "conversations to keep at once; the context is cut into that many slots, each holding its own")
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
	m, err := engine.Open(*model, *context, *parallel)
	if err != nil {
		fail(err)
	}
	defer m.Close()

	params := m.Sampling
	params.Seed = rand.Uint64()

	// One goroutine owns the model; every conversation asks it for its passes,
	// and what is waiting at the moment a pass is built goes into it together.
	runner := NewRunner(m.Forward)
	stop := make(chan struct{})
	defer close(stop)
	go runner.Run(stop)

	// One generator per slot: they share the weights, the vocabulary and the
	// count of calls handed out, and differ only in which cache they write.
	calls := 0
	slots := make([]*slot, m.Slots())
	for i := range slots {
		ctx := NewSlotContext(runner, i, m.Window, m.SlotContext(), time.Now, *ttl)
		gen := NewGenerator(ctx, m.Vocab, m.Template, m.Vocabulary, *maxTokens)
		gen.calls = &calls
		slots[i] = &slot{index: i, ctx: ctx, gen: gen}
	}
	pool := NewPool(slots, time.Now)
	name := strings.TrimSuffix(filepath.Base(*model), ".gguf")
	server := NewServer(pool, m.Vocab, name, m.Template, params)

	fmt.Fprintf(os.Stderr, "%s: %s, %d blocks, %d positions in %d slot(s) of %d, loaded in %s on %d cores\n",
		name, m.Name, m.Blocks, *context, m.Slots(), m.SlotContext(),
		time.Since(start).Round(time.Millisecond), runtime.NumCPU())
	// The port is taken before it is announced: an address already in use must
	// not be reported as a server that is listening.
	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		fail(err)
	}
	fmt.Fprintf(os.Stderr, "listening on http://%s/v1 — %d conversations at once, batched into one pass\n",
		listener.Addr(), m.Slots())
	if err := http.Serve(listener, logging(os.Stderr, server.Handler())); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "golem-server:", err)
	os.Exit(1)
}
