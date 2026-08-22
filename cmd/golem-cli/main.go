// Command golem-cli holds a conversation with a GGUF model, on the CPU.
//
//	golem-cli -model gemma-4-E2B-it-QAT-Q4_0.gguf
//	golem-cli -model Qwen3-4B-Q4_0.gguf -p "Explain a mutex in one sentence." -stats
//
// Which engine reads the file is read from the file: it declares its own
// architecture, and gemma4 and qwen3 are the two that are implemented.
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

	"github.com/ThiraSoft/golem/engine"
)

// stringList collects a flag given more than once, in the order it was given.
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ", ") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

// readAll reads the named files, still encoded: what decodes a picture is the
// engine that is about to look at it.
func readAll(paths []string) ([][]byte, error) {
	if len(paths) == 0 {
		return nil, nil
	}
	out := make([][]byte, len(paths))
	for i, path := range paths {
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		out[i] = raw
	}
	return out, nil
}

func main() {
	model := flag.String("model", os.Getenv("GOLEM_MODEL"), "GGUF file, gemma4 or qwen3 (or GOLEM_MODEL)")
	context := flag.Int("context", 4096, "positions to keep; the files declare far more than any machine here would survive")
	system := flag.String("system", "", "system message opening the conversation")
	think := flag.Bool("think", false, "open the system turn with the thinking marker")
	maxTokens := flag.Int("n", 512, "most tokens to draw for one answer")
	temp := flag.Float64("temp", -1, "temperature; 0 is greedy, negative takes the file's own value")
	topK := flag.Int("top-k", -1, "candidates kept; 0 keeps all, negative takes the file's own value")
	topP := flag.Float64("top-p", -1, "share of the mass kept; negative takes the file's own value")
	seed := flag.Uint64("seed", 0, "seed of the draw; 0 for a different answer every time")
	mmproj := flag.String("mmproj", os.Getenv("GOLEM_MMPROJ"), "projector GGUF, which is what lets a model see (or GOLEM_MMPROJ)")
	var images stringList
	flag.Var(&images, "image", "a picture to put in the first turn; repeat for several")
	var recordings stringList
	flag.Var(&recordings, "audio", "a sound file — WAV, MP3 or FLAC — to put in the first turn; repeat for several")
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
	if len(images) > 0 && *mmproj == "" {
		fail(fmt.Errorf("-image needs -mmproj: the projector is a separate file, and this model was opened without one"))
	}
	if len(recordings) > 0 && *mmproj == "" {
		fail(fmt.Errorf("-audio needs -mmproj: the projector is a separate file, and this model was opened without one"))
	}

	start := time.Now()
	m, err := engine.Open(*model, *context)
	if err != nil {
		fail(err)
	}
	defer m.Close()
	if *mmproj != "" {
		if err := m.OpenProjector(*mmproj); err != nil {
			fail(err)
		}
	}
	loading := time.Since(start)

	params := m.Sampling
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
		fmt.Fprintf(os.Stderr, "%s: %s, %d blocks, %d positions, vocabulary %d, loaded in %s on %d cores\n",
			filepath.Base(*model), m.Name, m.Blocks, *context, m.Vocabulary,
			loading.Round(time.Millisecond), runtime.NumCPU())
		fmt.Fprintf(os.Stderr, "sampling: temperature %g, top-k %d, top-p %g, seed %d\n",
			params.Temperature, params.TopK, params.TopP, params.Seed)
	}

	newSession := func() *Session {
		s := NewSession(m.Forward, m.Vocab, m.Template, params,
			m.Vocabulary, *context, *maxTokens, *system, *think)
		if v, ok := m.Media(); ok {
			s.SetVision(v)
		}
		return s
	}
	session := newSession()
	out := bufio.NewWriter(os.Stdout)

	// The pictures and the recordings go with the first turn and are not
	// repeated after it: the conversation keeps them, and the model has
	// already looked and listened.
	pending, pendingAudio := images, recordings
	answer := func(text string) {
		raw, err := readAll(pending)
		if err != nil {
			fail(err)
		}
		heard, err := readAll(pendingAudio)
		if err != nil {
			fail(err)
		}
		pending, pendingAudio = nil, nil
		turn, err := session.AskWithMedia(text, raw, heard, &flushing{out})
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
			m.Forward.Reset()
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
	fmt.Fprintln(os.Stderr, "golem-cli:", err)
	os.Exit(1)
}
