package gemma

// Reading what llama.cpp recorded. The fixtures are committed; a machine
// without them skips.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fixtureTensor struct {
	File string   `json:"file"`
	NE   [4]int64 `json:"ne"`
}

type logitProbe struct {
	ID    int32   `json:"id"`
	Logit float32 `json:"logit"`
}

type fixture struct {
	dir string
	// Model is the base name of the checkpoint the recording was made from.
	// The recorder has always written it; a fixture predating the field names
	// nothing and is accepted, so an old recording keeps working.
	Model     string                   `json:"model"`
	Tokens    []int32                  `json:"tokens"`
	NEmbd     int                      `json:"n_embd"`
	NLayer    int                      `json:"n_layer"`
	NPerLayer int                      `json:"n_embd_per_layer"`
	Argmax    int32                    `json:"argmax"`
	Greedy    []int32                  `json:"greedy"`
	LogitsTop []logitProbe             `json:"logits_top"`
	Tensors   map[string]fixtureTensor `json:"tensors"`
}

// repoRoot walks up to the module root, so a test does not care which package
// directory it was started from.
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

func loadFixture(t *testing.T, name string) *fixture {
	t.Helper()
	dir := filepath.Join(repoRoot(t), "testdata", "gemma", name)
	raw, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		t.Skipf("the %s fixtures are not on this machine: %v", name, err)
	}
	f := &fixture{}
	if err := json.Unmarshal(raw, f); err != nil {
		t.Fatalf("%s/index.json: %v", name, err)
	}
	f.dir = dir
	return f
}

// tensor reads one recording whole, in ggml's layout: dimension 0 fastest.
func (f *fixture) tensor(t *testing.T, name string) []float32 {
	t.Helper()
	e, ok := f.Tensors[name]
	if !ok {
		t.Fatalf("the fixture holds no tensor named %q", name)
	}
	raw, err := os.ReadFile(filepath.Join(f.dir, e.File))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw)%4 != 0 {
		t.Fatalf("%s is %d bytes, not a whole number of floats", e.File, len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	want := int(e.NE[0] * e.NE[1] * e.NE[2] * e.NE[3])
	if len(out) != want {
		t.Fatalf("%s holds %d floats, its shape says %d", e.File, len(out), want)
	}
	return out
}

// column returns element `index` of dimension 1 — one token's worth of a
// [n, tokens] recording.
func (f *fixture) column(t *testing.T, name string, index int) []float32 {
	t.Helper()
	e := f.Tensors[name]
	all := f.tensor(t, name)
	n := int(e.NE[0])
	if index < 0 || int64(index) >= e.NE[1] {
		t.Fatalf("%s has %d columns, asked for %d", name, e.NE[1], index)
	}
	return all[index*n : (index+1)*n]
}

// heads returns one token's worth of a head-shaped recording — a tensor of
// [head_dim, n_head, tokens], which is how llama.cpp names the queries, keys
// and values. The heads are contiguous, so this is the same concatenation the
// engine computes with.
func (f *fixture) heads(t *testing.T, name string, token int) []float32 {
	t.Helper()
	e := f.Tensors[name]
	all := f.tensor(t, name)
	n := int(e.NE[0] * e.NE[1])
	if token < 0 || int64(token) >= e.NE[2] {
		t.Fatalf("%s holds %d tokens, asked for %d", name, e.NE[2], token)
	}
	return all[token*n : (token+1)*n]
}

// compare fails with the worst element and its index. A maximum on its own says
// that something is wrong; the index says where to look.
func compare(t *testing.T, what string, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values against %d", what, len(got), len(want))
	}
	worst, at := float32(0), -1
	for i := range got {
		d := float32(math.Abs(float64(got[i] - want[i])))
		if d > worst {
			worst, at = d, i
		}
	}
	if worst > tolerance {
		t.Fatalf("%s: worst gap %g at element %d (%v against %v), tolerance %g",
			what, worst, at, got[at], want[at], tolerance)
	}
	t.Logf("%s: worst gap %g over %d values", what, worst, len(got))
}

// compareRelative is compare with the tolerance read against the size of what
// is being compared, which is what the attention outputs need.
//
// ggml hands the attention probabilities to its value product as fp16, whose
// step near a half is 5e-4. A score computed here differs from ggml's in the
// sixth digit — the dot products are summed in a different order — and a
// probability sitting within half a step of a boundary therefore rounds the
// other way now and then. When it does, the head moves by one step times the
// spread of its values, so the gap scales with the activation and an absolute
// tolerance would have to be set by the largest block and would say nothing
// about the smallest.
func compareRelative(t *testing.T, what string, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values against %d", what, len(got), len(want))
	}
	peak := float32(1)
	for _, v := range want {
		if a := float32(math.Abs(float64(v))); a > peak {
			peak = a
		}
	}
	worst, at := float32(0), -1
	for i := range got {
		d := float32(math.Abs(float64(got[i] - want[i])))
		if d > worst {
			worst, at = d, i
		}
	}
	if worst > tolerance*peak {
		t.Fatalf("%s: worst gap %g at element %d (%v against %v), %g of a peak of %v, tolerance %g",
			what, worst, at, got[at], want[at], worst/peak, peak, tolerance)
	}
	t.Logf("%s: worst gap %g over %d values, %g of a peak of %v",
		what, worst, len(got), worst/peak, peak)
}

func TestFixturesAreReadable(t *testing.T) {
	f := loadFixture(t, "layers")
	if len(f.Tokens) < 4 {
		t.Fatalf("%d tokens", len(f.Tokens))
	}
	if f.Tokens[0] != 2 {
		t.Fatalf("the first token is %d, expected the beginning-of-sequence 2", f.Tokens[0])
	}
	if f.NEmbd != 1536 || f.NLayer != 35 {
		t.Fatalf("n_embd %d, n_layer %d", f.NEmbd, f.NLayer)
	}
	if len(f.Greedy) != 16 || len(f.LogitsTop) != 64 {
		t.Fatalf("%d greedy tokens, %d logit probes", len(f.Greedy), len(f.LogitsTop))
	}

	// The embedding recording has one column per token, each of n_embd.
	e := f.Tensors["inp_scaled"]
	if int(e.NE[0]) != f.NEmbd || int(e.NE[1]) != len(f.Tokens) {
		t.Fatalf("inp_scaled has shape %v for %d tokens of %d", e.NE, len(f.Tokens), f.NEmbd)
	}
	if got := len(f.column(t, "inp_scaled", 0)); got != f.NEmbd {
		t.Fatalf("a column is %d wide", got)
	}

	// The per-layer input is one 256-vector per block per token.
	pl := f.Tensors["inp_per_layer"]
	if int(pl.NE[0]) != f.NPerLayer {
		t.Fatalf("inp_per_layer shape %v", pl.NE)
	}

	w := loadFixture(t, "window")
	if len(w.Tokens) <= 512 {
		t.Fatalf("the window fixture has only %d tokens", len(w.Tokens))
	}
	if int(w.Tensors["l_out-0"].NE[1]) != 1 {
		t.Fatal("the window fixture should keep only the last position")
	}
}

// The guard the 0.6B-for-4B mix-up asked for.
//
// Every recording names the checkpoint it was made from, and has since the
// recorder was written; nothing read the name back. So a variable pointing at
// the wrong file of the right family failed somewhere in the arithmetic —
// "activation length mismatch", from a function that knows nothing about
// checkpoints — instead of saying which file was wanted. These two say it.

// modelMismatch answers whether this fixture was recorded from a checkpoint
// other than the one about to be read. The comparison is on the base name
// alone: the recorder ran from a different directory than the test does.
func (f *fixture) modelMismatch(path string) error {
	if f.Model == "" || filepath.Base(path) == f.Model {
		return nil
	}
	return fmt.Errorf("these fixtures were recorded from %s, and %s was opened instead",
		f.Model, filepath.Base(path))
}

// requireModel fails the test rather than letting the mismatch reach the
// arithmetic.
func (f *fixture) requireModel(tb testing.TB, path string) {
	tb.Helper()
	if err := f.modelMismatch(path); err != nil {
		tb.Fatal(err)
	}
}

// TestFixturesNameTheirCheckpoint pairs each recording with the variable that
// names the file it came from. It is one test rather than a check inside every
// other, because the failure it catches is a misconfigured machine, not a
// misbehaving function, and a machine is misconfigured once.
func TestFixturesNameTheirCheckpoint(t *testing.T) {
	for _, c := range []struct{ dir, env string }{
		{"layers", "GOLEM_MODEL"},
		{"window", "GOLEM_MODEL"},
		{"layers12", "GOLEM_MODEL_12B"},
	} {
		path := os.Getenv(c.env)
		if path == "" {
			continue
		}
		f := loadFixture(t, c.dir)
		if err := f.modelMismatch(path); err != nil {
			t.Errorf("%s: %s names the wrong checkpoint: %v", c.dir, c.env, err)
		}
	}
}

// TestModelMismatchNamesTheFile pins what the message has to contain, because
// a guard that fires without saying what it wanted is the thing being fixed.
func TestModelMismatchNamesTheFile(t *testing.T) {
	f := &fixture{Model: "gemma-4-26B-A4B-it-QAT-Q4_0.gguf"}
	err := f.modelMismatch("/models/gemma-4-12B-it-QAT-Q4_0.gguf")
	if err == nil {
		t.Fatal("a recording from another checkpoint was accepted")
	}
	if !strings.Contains(err.Error(), "gemma-4-26B-A4B-it-QAT-Q4_0.gguf") {
		t.Fatalf("the message does not name the recording's own model: %v", err)
	}
	if err := f.modelMismatch("/elsewhere/gemma-4-26B-A4B-it-QAT-Q4_0.gguf"); err != nil {
		t.Fatalf("the matching checkpoint was rejected: %v", err)
	}
	if err := (&fixture{}).modelMismatch("/models/anything.gguf"); err != nil {
		t.Fatalf("a recording that names no model must be accepted: %v", err)
	}
}
