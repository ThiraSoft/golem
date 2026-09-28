// Command laya asks Laya typed questions about a state and prints the answers
// in the shape Jev returns them.
//
// The request is JSON, from a file or standard input:
//
//	{"state": {...} or "text", "questions": {"id": {"type": "choice", "instructions": "...", "criteria": [...]}}}
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/ThiraSoft/golem/internal/version"
	"github.com/ThiraSoft/golem/laya"
)

func main() {
	model := flag.String("model", os.Getenv("GOLEM_MODEL_LAYA"),
		"the checkpoint directory as Hugging Face ships it (convaiinnovations/laya)")
	vulkan := flag.Bool("vulkan", false, "run on the GPU through Vulkan")
	stats := flag.Bool("stats", false, "print the time the pass took on standard error")
	showVersion := flag.Bool("version", false, "print the version and exit")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s -model <dir> [options] [request.json]\n", filepath.Base(os.Args[0]))
		fmt.Fprintln(os.Stderr, "  Without a file, the request is read from standard input.")
		flag.PrintDefaults()
	}
	flag.Parse()
	if *showVersion {
		fmt.Println(filepath.Base(os.Args[0]), version.String())
		return
	}
	if *model == "" {
		fail(fmt.Errorf("no model: give -model or set GOLEM_MODEL_LAYA"))
	}

	var raw []byte
	var err error
	if flag.NArg() > 0 {
		raw, err = os.ReadFile(flag.Arg(0))
	} else {
		raw, err = io.ReadAll(os.Stdin)
	}
	if err != nil {
		fail(err)
	}
	var req struct {
		State     json.RawMessage `json:"state"`
		Questions json.RawMessage `json:"questions"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		fail(fmt.Errorf("request: %w", err))
	}
	if len(req.State) == 0 || len(req.Questions) == 0 {
		fail(fmt.Errorf("the request needs a state and questions"))
	}
	questions, err := laya.ParseQuestions(req.Questions)
	if err != nil {
		fail(err)
	}

	m, err := laya.Open(*model)
	if err != nil {
		fail(err)
	}
	defer m.Close()
	if *vulkan {
		if err := m.UseVulkan(); err != nil {
			fail(err)
		}
	}
	start := time.Now()
	res, err := m.Decide(context.Background(), req.State, questions)
	if err != nil {
		fail(err)
	}
	if *stats {
		fmt.Fprintf(os.Stderr, "%d questions, %d positions, %v\n", len(questions), res.InputTokens, time.Since(start).Round(time.Millisecond))
	}
	out, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		fail(err)
	}
	fmt.Println(string(out))
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "laya:", err)
	os.Exit(1)
}
