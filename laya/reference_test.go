package laya

// Parity with PyTorch, waypoint by waypoint: ref/laya/dump.py runs the
// checkpoint's own code on the processor in float32 and records every block's
// output, for each question of two requests, into testdata/laya.

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
)

type refCase struct {
	Name     string          `json:"name"`
	State    json.RawMessage `json:"state"`
	Question json.RawMessage `json:"question"`
	IDs      []int32         `json:"ids"`
	Markers  []int           `json:"markers"`
	Answer   json.RawMessage `json:"answer"`
}

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found above the current directory")
		}
		dir = parent
	}
}

func openModel(t *testing.T) *Model {
	t.Helper()
	dir := os.Getenv("GOLEM_MODEL_LAYA")
	if dir == "" {
		t.Skip("GOLEM_MODEL_LAYA is not set")
	}
	m, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func loadCases(t *testing.T) (string, []refCase) {
	t.Helper()
	dir := filepath.Join(repoRoot(t), "testdata", "laya")
	raw, err := os.ReadFile(filepath.Join(dir, "cases.json"))
	if err != nil {
		t.Skip("the laya fixtures are not on this machine")
	}
	var cases []refCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	return dir, cases
}

func readFloats(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

// TestTokenizerMatchesHuggingFace holds the checkpoint's tokenizer, whichever
// kind it is, to what Hugging Face's gives the texts ref/laya/dump.py records.
func TestTokenizerMatchesHuggingFace(t *testing.T) {
	m := openModel(t)
	dir, _ := loadCases(t)
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		if got := m.Vocab.Encode(c.Text, false, true); !slices.Equal(got, c.IDs) {
			t.Errorf("%q:\n got %v\nwant %v", c.Text, got, c.IDs)
		}
	}
}

// TestSequencesMatchPyTorch holds build_sequence, the state's serialization
// and the tokenizer together to what the reference fed its model.
func TestSequencesMatchPyTorch(t *testing.T) {
	m := openModel(t)
	_, cases := loadCases(t)
	for _, c := range cases {
		var q Question
		if err := json.Unmarshal(c.Question, &q); err != nil {
			t.Fatal(err)
		}
		p, err := prepare(q)
		if err != nil {
			t.Fatal(err)
		}
		text, err := StateText(c.State)
		if err != nil {
			t.Fatal(err)
		}
		ids, markers := m.sequence(text, p)
		if !slices.Equal(ids, c.IDs) {
			t.Errorf("%s: ids\n got %v\nwant %v", c.Name, ids, c.IDs)
		}
		if !slices.Equal(markers, c.Markers) {
			t.Errorf("%s: markers %v, want %v", c.Name, markers, c.Markers)
		}
	}
}

func order(name string) int {
	base, idx, _ := strings.Cut(name, "-")
	i, _ := strconv.Atoi(idx)
	switch base {
	case "embed":
		return 0
	case "layer":
		return 1 + i
	case "encoded":
		return 100
	case "head":
		return 101 + i
	}
	return 200
}

func TestWaypointsMatchPyTorch(t *testing.T) {
	m := openModel(t)
	dir, cases := loadCases(t)
	checkWaypoints(t, m, dir, cases, 2e-4)
}

func checkWaypoints(t *testing.T, m *Model, dir string, cases []refCase, tolerance float64) {
	for _, c := range cases {
		var q Question
		if err := json.Unmarshal(c.Question, &q); err != nil {
			t.Fatal(err)
		}
		p, err := prepare(q)
		if err != nil {
			t.Fatal(err)
		}
		got := map[string][]float32{}
		m.trace = func(name string, rows [][]float32) {
			var flat []float32
			for _, r := range rows {
				flat = append(flat, r...)
			}
			got[name] = flat
		}
		outs, err := m.forward([]seq{{ids: c.IDs, markers: c.Markers, kind: p.kind}})
		m.trace = nil
		if err != nil {
			t.Fatal(err)
		}
		got["act"] = outs[0].actLogits[:]

		var shapes map[string][]int
		raw, err := os.ReadFile(filepath.Join(dir, c.Name, "index.json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &shapes); err != nil {
			t.Fatal(err)
		}
		names := make([]string, 0, len(shapes))
		for n := range shapes {
			names = append(names, n)
		}
		sort.Slice(names, func(a, b int) bool { return order(names[a]) < order(names[b]) })
		worst, worstName := 0.0, ""
		for _, name := range names {
			want := readFloats(t, filepath.Join(dir, c.Name, name+".bin"))
			have, ok := got[name]
			if !ok {
				if m.gpu == nil {
					t.Errorf("%s/%s: never produced", c.Name, name)
				}
				continue
			}
			if len(have) != len(want) {
				t.Errorf("%s/%s: %d values, want %d", c.Name, name, len(have), len(want))
				continue
			}
			var peak, diff float64
			for i := range want {
				peak = math.Max(peak, math.Abs(float64(want[i])))
				diff = math.Max(diff, math.Abs(float64(have[i]-want[i])))
			}
			rel := diff / math.Max(peak, 1e-30)
			if rel > worst {
				worst, worstName = rel, name
			}
			if rel > tolerance {
				t.Errorf("%s/%s: max|Δ| %.3g peak %.3g rel %.3g above %.3g", c.Name, name, diff, peak, rel, tolerance)
			}
		}
		t.Logf("%-18s worst %-10s rel %.3g", c.Name, worstName, worst)
	}
}

// TestDecideMatchesSystemOne asks each request as the reference asked it, all
// its questions in one pass, and compares the calibrated answers.
func TestDecideMatchesSystemOne(t *testing.T) {
	m := openModel(t)
	_, cases := loadCases(t)
	checkDecide(t, m, cases, 1e-5)
}

func checkDecide(t *testing.T, m *Model, cases []refCase, tolerance float64) {
	groups := map[string][]refCase{}
	var keys []string
	for _, c := range cases {
		g, _, _ := strings.Cut(c.Name, "_")
		if _, ok := groups[g]; !ok {
			keys = append(keys, g)
		}
		groups[g] = append(groups[g], c)
	}
	for _, g := range keys {
		cs := groups[g]
		var qs []Named
		for _, c := range cs {
			var q Question
			json.Unmarshal(c.Question, &q)
			_, id, _ := strings.Cut(c.Name, "_")
			qs = append(qs, Named{ID: id, Question: q})
		}
		res, err := m.Decide(context.Background(), cs[0].State, qs)
		if err != nil {
			t.Fatal(err)
		}
		for i, c := range cs {
			var want struct {
				Choice        string             `json:"choice"`
				Score         float64            `json:"score"`
				Noul          float64            `json:"noul"`
				Probabilities map[string]float64 `json:"probabilities"`
				Confidence    float64            `json:"confidence"`
			}
			json.Unmarshal(c.Answer, &want)
			a := res.Answers[i]
			gotJSON, _ := json.Marshal(a)
			t.Logf("%s: %s", c.Name, gotJSON)
			if a.Label != want.Choice {
				t.Errorf("%s: choice %q, want %q", c.Name, a.Label, want.Choice)
			}
			near := func(what string, got, want float64) {
				if math.Abs(got-want) > tolerance+5e-5 {
					t.Errorf("%s: %s %.5f, want %.4f", c.Name, what, got, want)
				}
			}
			near("score", a.Score, want.Score)
			near("noul", a.Noul, want.Noul)
			near("confidence", a.Confidence, want.Confidence)
			for _, p := range a.Probabilities {
				near("p("+p.Label+")", p.P, want.Probabilities[p.Label])
			}
		}
	}
}

// TestVulkanMatchesPyTorch is the same comparison on the card, whose products
// meet the matrix cores in fp16 halves and whose sums go in another order.
// The bar is looser than the processor's by the long fixture's position 209,
// which ModernBERT turns into a massive activation at block 19 and which
// amplifies whatever reaches it.
func TestVulkanMatchesPyTorch(t *testing.T) {
	m := openModel(t)
	dir, cases := loadCases(t)
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no card: %v", err)
	}
	checkWaypoints(t, m, dir, cases, 2e-3)
	checkDecide(t, m, cases, 5e-4)
}
