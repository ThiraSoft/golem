package main

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ThiraSoft/golem/engine"
)

// What ref/dump_logprobs.cpp recorded: closed questions rendered by Gemma 4's
// template, and llama.cpp's log-probability of each option, token by token.
type logprobCase struct {
	Name    string  `json:"name"`
	Prompt  []int32 `json:"prompt"`
	Options []struct {
		Text     string    `json:"text"`
		Tokens   []int32   `json:"tokens"`
		LogProbs []float64 `json:"logprobs"`
		Sum      float64   `json:"sum"`
	} `json:"options"`
}

// The options are weighed as llama.cpp weighs them: E2B on the processor, the
// card having no per-layer branch for it, and the 12B on the card, which is
// where a server runs it. ref/systemone/README.md says how the fixtures are
// recorded and what the gaps were.
func TestWeighMatchesLlamaCpp(t *testing.T) {
	for _, run := range []struct {
		name, env, fixture string
		card               bool
		nats, p            float64
	}{
		{"e2b/processor", "GOLEM_MODEL", "e2b", false, 0.3, 0.05},
		{"12b/card", "GOLEM_MODEL_12B", "12b", true, 1.2, 0.05},
	} {
		t.Run(run.name, func(t *testing.T) {
			path := os.Getenv(run.env)
			if path == "" {
				t.Skipf("%s is not set", run.env)
			}
			raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "systemone", run.fixture, "logprobs.json"))
			if err != nil {
				t.Skipf("the fixture is not on this machine (ref/systemone/README.md): %v", err)
			}
			var fixture struct{ Cases []logprobCase }
			if err := json.Unmarshal(raw, &fixture); err != nil {
				t.Fatal(err)
			}
			m, err := engine.Open(path, 2048)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if run.card {
				if err := m.UseVulkan(); err != nil {
					t.Skipf("no card: %v", err)
				}
			}
			r := running(t, m.Forward)
			if _, blocks := m.Vulkan(); blocks {
				r.OnDevice()
			}
			c := NewContext(r, m.Window, 2048, time.Now, 0)
			c.SetRing(m.WindowRing())
			text := func(ids []int32) (engine.Prompt, error) { return engine.TextPrompt(ids), nil }
			open, close := m.Template.ReasoningMarkers()
			lead := m.Vocab.Encode(open+"\n"+close, false, true)
			worst, worstP := 0.0, 0.0
			for _, tc := range fixture.Cases {
				// A fixture recorded past the empty thought is weighed without
				// it: Weigh has to decide to write it, as the server does.
				prompt := tc.Prompt
				if base, ok := bytesCut(prompt, lead); ok {
					prompt = base
				}
				n := len(prompt)
				w := weighing{tail: prompt[n-1:]}
				for _, o := range tc.Options {
					w.options = append(w.options, o.Tokens)
				}
				got, _, err := c.Weigh(context.Background(), text, prompt[:n-1], []weighing{w}, lead, m.Vocabulary)
				if err != nil {
					t.Fatal(err)
				}
				var want []float64
				for i, o := range tc.Options {
					want = append(want, o.Sum)
					worst = math.Max(worst, math.Abs(got[0][i]-o.Sum))
				}
				p, q := distribution(got[0]), distribution(want)
				for i := range p {
					worstP = math.Max(worstP, math.Abs(p[i]-q[i]))
				}
				if slices.Index(p, slices.Max(p)) != slices.Index(q, slices.Max(q)) {
					t.Errorf("%s: golem picks another option: %v against %v", tc.Name, p, q)
				}
			}
			t.Logf("worst gap %.4f nats in an option's log-probability, %.4f in its probability", worst, worstP)
			if worst > run.nats || worstP > run.p {
				t.Errorf("worst gap %.4f nats, %.4f in probability: past %.2f and %.2f", worst, worstP, run.nats, run.p)
			}
		})
	}
}

// bytesCut is ids without suffix, if it ends with it.
func bytesCut(ids, suffix []int32) ([]int32, bool) {
	if len(ids) < len(suffix) || !slices.Equal(ids[len(ids)-len(suffix):], suffix) {
		return ids, false
	}
	return ids[:len(ids)-len(suffix)], true
}

// golem reads the fixture's prompts into the tokens llama.cpp read.
func TestWeighFixtureTokensAreGolems(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL")
	if path == "" {
		t.Skip("GOLEM_MODEL is not set")
	}
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "systemone", "e2b", "logprobs.json"))
	if err != nil {
		t.Skipf("the fixture is not on this machine (ref/systemone/README.md): %v", err)
	}
	var fixture struct{ Cases []logprobCase }
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	tsv, err := os.ReadFile(filepath.Join("..", "..", "ref", "systemone", "cases.tsv"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := engine.Open(path, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for i, line := range corpusLines(string(tsv)) {
		prompt := m.Vocab.Encode(line[1], false, true)
		if !slices.Equal(prompt, fixture.Cases[i].Prompt) {
			t.Fatalf("%s: golem reads the prompt into other tokens", line[0])
		}
		for j, o := range line[2:] {
			ids := m.Vocab.Encode(line[1]+o, false, true)
			if !slices.Equal(ids[len(prompt):], fixture.Cases[i].Options[j].Tokens) {
				t.Fatalf("%s: option %q is %v, llama.cpp read %v", line[0], o, ids[len(prompt):], fixture.Cases[i].Options[j].Tokens)
			}
		}
	}
}

// corpusLines reads ref/systemone/cases.tsv: tab-separated columns, comments
// and blank lines left out, dump_tokens' escapes undone.
func corpusLines(tsv string) [][]string {
	var out [][]string
	for _, line := range strings.Split(tsv, "\n") {
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		cols := strings.Split(line, "\t")
		for i, c := range cols {
			cols[i] = strings.NewReplacer(`\\`, `\`, `\n`, "\n", `\t`, "\t", `\r`, "\r", `\s`, " ").Replace(c)
		}
		out = append(out, cols)
	}
	return out
}
