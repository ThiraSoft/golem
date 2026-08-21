package qwen

// Reading what llama.cpp recorded. The fixtures are committed; a machine
// without them skips.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
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
	dir       string
	Tokens    []int32                  `json:"tokens"`
	NEmbd     int                      `json:"n_embd"`
	NLayer    int                      `json:"n_layer"`
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
	dir := filepath.Join(repoRoot(t), "testdata", "qwen", name)
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

// lastColumn returns the final column of a recording, whatever its width.
//
// The last block's waypoints hold one column rather than one per token:
// llama.cpp narrows the graph to the output row before block 27's feed
// forward, because that row is the only one whose logits anyone wants. So a
// test that wants "the last position" has to ask for the last column rather
// than for column len(tokens)-1.
func (f *fixture) lastColumn(t *testing.T, name string) []float32 {
	t.Helper()
	e, ok := f.Tensors[name]
	if !ok {
		t.Fatalf("the fixture holds no tensor named %q", name)
	}
	return f.column(t, name, int(e.NE[1])-1)
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

	// Five tokens, not six: this tokenizer prepends nothing, and a recording
	// that began with a marker would be a recording of another prompt.
	if len(f.Tokens) != 5 {
		t.Fatalf("%d tokens, want 5", len(f.Tokens))
	}
	if f.NEmbd != 1024 || f.NLayer != 28 {
		t.Fatalf("n_embd %d, n_layer %d", f.NEmbd, f.NLayer)
	}
	if len(f.Greedy) != 16 || len(f.LogitsTop) != 64 {
		t.Fatalf("%d greedy tokens, %d logit probes", len(f.Greedy), len(f.LogitsTop))
	}

	// The queries are recorded head-shaped: [head_dim, n_head, tokens].
	e, ok := f.Tensors["Qcur-0"]
	if !ok {
		t.Fatal("the fixture holds no Qcur-0")
	}
	if e.NE[0] != 128 || e.NE[1] != 16 || int(e.NE[2]) != len(f.Tokens) {
		t.Fatalf("Qcur-0 has shape %v", e.NE)
	}
	// And the keys with half as many heads, which is the whole of grouped
	// query attention as far as the shapes are concerned.
	if k := f.Tensors["Kcur-0"]; k.NE[1] != 8 {
		t.Fatalf("Kcur-0 has %d heads, want 8", k.NE[1])
	}

	// Gemma scales its embedding and records the result; this model does not
	// scale, so the waypoint does not exist. Asserting its absence is what
	// keeps someone from adding a scale later and finding nothing complains.
	if _, ok := f.Tensors["inp_scaled"]; ok {
		t.Error("the fixture has an inp_scaled, so this model does scale its embedding after all")
	}
}

func itoa(i int) string { return strconv.Itoa(i) }
