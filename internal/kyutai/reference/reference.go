package reference

// Access to the reference activations produced by the scripts in ref/.
//
// This package is used only by tests, but it is not a test file: several
// packages compare their outputs against the same fixtures, and the reading
// code has no reason to be copied into each of them.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Fixtures describes a set of activations and its geometry.
type Fixtures struct {
	Language    string         `json:"language"`
	NumLayers   int            `json:"num_layers"`
	DModel      int            `json:"d_model"`
	NumHeads    int            `json:"num_heads"`
	DimFF       int            `json:"dim_feedforward"`
	SeqLen      int            `json:"seq_len"`
	MaxPeriod   float64        `json:"max_period"`
	Frames      int            `json:"frames"`
	Temp        float64        `json:"temp"`
	EOSThresh   float64        `json:"eos_threshold"`
	VoiceOffset int            `json:"voice_offset"`
	Text        string         `json:"text"`
	Tensors     map[string]any `json:"tensors"`
	dir         string
}

// PipelineSets are the end-to-end fixture sets, one per language exercised.
// Adding a language means adding its directory here and nothing else.
var PipelineSets = []string{"pipeline", "pipeline_en"}

// Load reads the named fixture set, or skips the test if it is absent.
func Load(t *testing.T, set string) Fixtures {
	t.Helper()
	dir := filepath.Join(root(t), "testdata", set)
	raw, err := os.ReadFile(filepath.Join(dir, "fixtures.json"))
	if err != nil {
		t.Skipf("fixtures %q missing (%v) — see ref/", set, err)
	}
	var f Fixtures
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	f.dir = dir
	if f.Language == "" {
		f.Language = DefaultLanguage
	}
	return f
}

// Read returns the named activation, flattened.
func (f Fixtures) Read(t *testing.T, name string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(f.dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// Compare checks the maximum relative gap between two vectors.
func Compare(t *testing.T, name string, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	var worstGap float32
	var worstIndex int
	var norm float64
	for _, v := range want {
		norm += float64(v) * float64(v)
	}
	scale := float32(math.Sqrt(norm/float64(len(want)))) + 1e-6
	for i := range got {
		gap := float32(math.Abs(float64(got[i] - want[i])))
		if gap > worstGap {
			worstGap, worstIndex = gap, i
		}
	}
	relative := worstGap / scale
	if relative > tolerance {
		t.Errorf("%s: gap %.3g at index %d (got %.6f, want %.6f), that is %.2f%% of the scale — beyond %.2f%%",
			name, worstGap, worstIndex, got[worstIndex], want[worstIndex], relative*100, tolerance*100)
		return
	}
	t.Logf("%s: max gap %.3g (%.4f%% of the scale)", name, worstGap, relative*100)
}

// Ints returns a file of 32-bit integers, as the tokenizer produces.
func (f Fixtures) Ints(t *testing.T, name string) []int {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(f.dir, name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int, len(raw)/4)
	for i := range out {
		out[i] = int(int32(binary.LittleEndian.Uint32(raw[i*4:])))
	}
	return out
}

// root walks up to the go.mod: the tests run from their own package directory,
// at differing depths.
func root(t testing.TB) string {
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
