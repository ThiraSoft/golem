// Command golem-server answers an OpenAI-compatible API over a GGUF, on the
// CPU or, with -vulkan, with the model's blocks and logit head on a Vulkan
// device.
//
//	golem-server -model gemma-4-E2B-it-QAT-Q4_0.gguf -addr :8080
//	golem-server -model Qwen3-4B-Q4_0.gguf -addr :8080
//
// Which engine reads the file is read from the file: it declares its own
// architecture, and gemma4, qwen3 and qwen35 — which is Qwen3.8 — are the three
// that are implemented. It serves /v1/chat/completions, streamed or not, tool
// declarations included, and /v1/models. It reports the calls the model makes;
// running them is the client's part, as the protocol has it.
package main

import (
	"context"
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
	"github.com/ThiraSoft/golem/internal/version"
	"github.com/ThiraSoft/golem/sample"
	"github.com/ThiraSoft/golem/stt"
)

func main() {
	model := flag.String("model", os.Getenv("GOLEM_MODEL"), "GGUF file, gemma4, qwen3 or qwen35 (or GOLEM_MODEL)")
	sttDir := flag.String("stt", os.Getenv("GOLEM_STT"), "directory holding a Kyutai STT checkpoint (or GOLEM_STT)")
	mmproj := flag.String("mmproj", os.Getenv("GOLEM_MMPROJ"), "projector GGUF, which is what lets a model see (or GOLEM_MMPROJ)")
	addr := flag.String("addr", "127.0.0.1:8080", "address to listen on")
	context := flag.Int("context", 4096, "positions to keep; the files declare far more than any machine here would survive")
	maxTokens := flag.Int("n", 1024, "most tokens to draw for one answer, when the request names no limit")
	parallel := flag.Int("parallel", 1, "conversations to keep at once; the context is cut into that many slots, each holding its own")
	ttl := flag.Duration("cache-ttl", 0, "forget a conversation's tokens after this long idle; 0 never forgets. The memory is allocated at startup and is released by neither")
	vulkan := flag.Bool("vulkan", false, "put the logit head and the expert stacks on a Vulkan device")
	sttStreams := flag.Int("stt-parallel", 1, "transcriptions to carry at once; they are stepped together, so the trunk's weights are read once for all of them")
	showVersion := flag.Bool("version", false, "print the version and exit")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s [options]\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()

	if *showVersion {
		fmt.Println(filepath.Base(os.Args[0]), version.String())
		return
	}

	if *model == "" && *sttDir == "" {
		fail(fmt.Errorf("no model: pass -model or -stt, or set GOLEM_MODEL or GOLEM_STT"))
	}
	// A transcriber alone is a whole server. It holds no conversation, so
	// nothing below this — the runner, the slots, the generators — has anything
	// to own, and building them around a model that was never opened would only
	// be a longer way of writing nil.
	if *model == "" {
		serveTranscriptionsOnly(*sttDir, *addr, *sttStreams, *vulkan)
		return
	}
	start := time.Now()
	m, err := engine.Open(*model, *context, *parallel)
	if err != nil {
		fail(err)
	}
	defer m.Close()
	if *vulkan {
		if err := m.UseVulkan(); err != nil {
			fail(err)
		}
	}

	params := m.Sampling
	params.Seed = rand.Uint64()

	// One goroutine owns the model; every conversation asks it for its passes,
	// and what is waiting at the moment a pass is built goes into it together.
	if *mmproj != "" {
		if err := m.OpenProjector(*mmproj); err != nil {
			fail(err)
		}
	}
	runner := NewRunner(m.Forward)
	// A card reads a prompt at five times the rate the processor's width gets
	// out of it; context.go's devicePassWidth says the measurement.
	if _, blocks := m.Vulkan(); blocks {
		runner.OnDevice()
	}
	if v, ok := m.Media(); ok {
		runner.SetVision(v)
	}
	// The checkpoint's own prediction block, when it carries one and the card
	// holds the blocks it needs. It draws a second token out of the same
	// reading of the weights, for a conversation drawing alone; two of them are
	// better served by the pass that carries both, and Runner.CanDraft is what
	// weighs the two. qwen35/speculate.go says what the bargain is.
	drafting := false
	if d, ok := m.Forward.(drafter); ok && d.Speculate() {
		sp, err := d.NewSpeculator()
		if err != nil {
			fail(err)
		}
		runner.UseDrafter(sp, d.ResetMTP)
		drafting = true
	}
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
	if v, ok := m.Media(); ok {
		server.SetVision(v)
	}
	if *sttDir != "" {
		opts, err := stt.Locate(*sttDir)
		if err != nil {
			fail(err)
		}
		sttModel, err := stt.Open(opts)
		if err != nil {
			fail(err)
		}
		defer sttModel.Close()
		sttOnVulkan(sttModel, *vulkan, *sttStreams)
		server.SetSTT(sttModel)
		if group := sttGroupFor(sttModel, *sttStreams); group != nil {
			defer group.Close()
			server.SetSTTGroup(group)
		}
	}

	head := vulkanLine(m.Vulkan())
	draft := ""
	if drafting {
		draft = ", drafting with the prediction block"
	}
	fmt.Fprintf(os.Stderr, "%s: %s, %d blocks, %d positions in %d slot(s) of %d, %s%s, loaded in %s on %d cores\n",
		name, m.Name, m.Blocks, *context, m.Slots(), m.SlotContext(), head, draft,
		time.Since(start).Round(time.Millisecond), runtime.NumCPU())

	// The image tower, when there is one. It is worth a line of its own: a
	// tower that did not fit beside the model still runs on the card, and the
	// difference is a tenth of a second an image rather than a wrong answer.
	if on, resident := m.VisionVulkan(); on {
		how := "one group of weights at a time across the bus"
		if resident {
			how = "resident"
		}
		fmt.Fprintf(os.Stderr, "image tower on vulkan, %s\n", how)
	}

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

// sttGroupFor is the group that carries more than one transcription at once, or
// nil for a server carrying one: a group of one is a gathering window and a
// goroutine around a stream that would step alone anyway.
//
// It is a function and not four lines in main because main names a flag
// `context`, which hides the package of the same name for the rest of it.
func sttGroupFor(m *stt.Model, streams int) *stt.Group {
	if streams <= 1 {
		return nil
	}
	return m.Group(context.Background(), streams)
}

// sttOnVulkan puts the trunk's four products on the card, at the width the
// group will run. It fails loudly rather than falling back: a server told
// -vulkan and left on the processor would be a capacity plan built on a flag
// that did nothing.
func sttOnVulkan(m *stt.Model, vulkan bool, streams int) {
	if !vulkan {
		return
	}
	if streams < 1 {
		streams = 1
	}
	if err := m.UseVulkan(streams); err != nil {
		fail(fmt.Errorf("stt on vulkan: %w", err))
	}
	fmt.Fprintf(os.Stderr, "stt trunk on vulkan in %s, passes of %d\n", m.Quant(), streams)
}

// serveTranscriptionsOnly runs a server carrying an STT and nothing else.
// Server.Handler registers the conversation route only when there is a pool,
// so what this listens on is /v1/models and /v1/audio/transcriptions.
func serveTranscriptionsOnly(dir, addr string, streams int, vulkan bool) {
	start := time.Now()
	opts, err := stt.Locate(dir)
	if err != nil {
		fail(err)
	}
	model, err := stt.Open(opts)
	if err != nil {
		fail(err)
	}
	defer model.Close()

	name := filepath.Base(filepath.Clean(dir))
	server := NewServer(nil, nil, name, nil, sample.Params{})
	sttOnVulkan(model, vulkan, streams)
	server.SetSTT(model)
	if group := sttGroupFor(model, streams); group != nil {
		defer group.Close()
		server.SetSTTGroup(group)
	}

	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fail(err)
	}
	fmt.Fprintf(os.Stderr, "%s: speech to text, trunk in %s, loaded in %s on %d cores\n",
		name, model.Quant(), time.Since(start).Round(time.Millisecond), runtime.NumCPU())
	carrying := "one at a time"
	if streams > 1 {
		carrying = fmt.Sprintf("%d at a time, stepped together", streams)
	}
	fmt.Fprintf(os.Stderr, "listening on http://%s/v1 — transcriptions only, %s\n", listener.Addr(), carrying)
	if err := http.Serve(listener, logging(os.Stderr, server.Handler())); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "golem-server:", err)
	os.Exit(1)
}

// vulkanLine names what ended up on the card, for the startup line.
func vulkanLine(head, blocks bool) string {
	switch {
	case head && blocks:
		return "blocks and head on vulkan"
	case blocks:
		return "blocks on vulkan"
	case head:
		return "head on vulkan"
	}
	return "all on cpu"
}
