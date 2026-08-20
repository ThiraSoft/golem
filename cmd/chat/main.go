// Command chat holds a conversation with Gemma 4, from a GGUF file, on the CPU.
//
//	chat -model gemma-4-E2B-it-QAT-Q4_0.gguf
//	chat -model … -p "Explain a mutex in one sentence." -stats
//
// With -p it answers once and exits; without, it reads turns from the terminal
// until end of file.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand/v2"
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
	context := flag.Int("context", 4096, "positions to keep; the file declares 131072, which no machine here would survive")
	system := flag.String("system", "", "system message opening the conversation")
	think := flag.Bool("think", false, "open the system turn with the thinking marker")
	maxTokens := flag.Int("n", 512, "most tokens to draw for one answer")
	temp := flag.Float64("temp", -1, "temperature; 0 is greedy, negative takes the file's own value")
	topK := flag.Int("top-k", -1, "candidates kept; 0 keeps all, negative takes the file's own value")
	topP := flag.Float64("top-p", -1, "share of the mass kept; negative takes the file's own value")
	seed := flag.Uint64("seed", 0, "seed of the draw; 0 for a different answer every time")
	prompt := flag.String("p", "", "answer this and exit, instead of reading turns")
	stats := flag.Bool("stats", false, "report tokens and speed after each answer")
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
	loading := time.Since(start)

	params := m.Cfg.Sampling
	if *temp >= 0 {
		params.Temperature = float32(*temp)
	}
	if *topK >= 0 {
		params.TopK = *topK
	}
	if *topP >= 0 {
		params.TopP = float32(*topP)
	}
	params.Seed = *seed
	if params.Seed == 0 {
		params.Seed = rand.Uint64()
	}

	if *stats {
		fmt.Fprintf(os.Stderr, "%s: %d blocks, %d positions, vocabulary %d, loaded in %s on %d cores\n",
			filepath.Base(*model), len(m.Cfg.Blocks), *context, m.Cfg.Vocab,
			loading.Round(time.Millisecond), runtime.NumCPU())
		fmt.Fprintf(os.Stderr, "sampling: temperature %g, top-k %d, top-p %g, seed %d\n",
			params.Temperature, params.TopK, params.TopP, params.Seed)
	}

	newSession := func() *Session {
		return NewSession(m, vocab, params, m.Cfg.Vocab, *context, *maxTokens, *system, *think)
	}
	session := newSession()
	out := bufio.NewWriter(os.Stdout)

	answer := func(text string) {
		turn, err := session.Ask(text, &flushing{out})
		out.Flush()
		fmt.Println()
		if err != nil {
			fail(err)
		}
		if *stats {
			report(turn)
		}
	}

	if *prompt != "" {
		answer(*prompt)
		return
	}

	fmt.Fprintln(os.Stderr, "Type a message. /reset forgets the conversation, end of file leaves.")
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64<<10), 1<<20)
	for {
		fmt.Fprint(os.Stderr, "> ")
		if !in.Scan() {
			break
		}
		line := strings.TrimSpace(in.Text())
		switch {
		case line == "":
			continue
		case line == "/reset":
			m.Reset()
			session = newSession()
			fmt.Fprintln(os.Stderr, "(forgotten)")
			continue
		}
		answer(line)
	}
	if err := in.Err(); err != nil {
		fail(err)
	}
}

// flushing writes a token to the terminal as soon as it exists, rather than
// when the buffer happens to fill.
type flushing struct{ w *bufio.Writer }

func (f *flushing) Write(p []byte) (int, error) {
	n, err := f.w.Write(p)
	if err != nil {
		return n, err
	}
	return n, f.w.Flush()
}

func report(t Turn) {
	rate := func(n int, d time.Duration) float64 {
		if d <= 0 {
			return 0
		}
		return float64(n) / d.Seconds()
	}
	note := ""
	if t.Truncated {
		note = ", cut short"
	}
	fmt.Fprintf(os.Stderr, "%d prompt tokens in %s (%.1f/s), %d generated in %s (%.2f/s)%s\n",
		t.Prompt, t.Prefill.Round(time.Millisecond), rate(t.Prompt, t.Prefill),
		t.Generated, t.Decode.Round(time.Millisecond), rate(t.Generated, t.Decode), note)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "chat:", err)
	os.Exit(1)
}
