package nn

// Reads the fixtures ref/gemma/dump_quants.cpp recorded from ggml. They are
// versioned, so no C++ is needed at test time; when they are absent the tests
// skip, as the pocket-tts fixtures already do.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type quantFixture struct {
	Rows, Cols int
	Weights    []byte
	X, Y       []float32
}

type quantIndexEntry struct {
	Tensor  string `json:"tensor"`
	Type    string `json:"type"`
	Rows    int    `json:"rows"`
	Cols    int    `json:"cols"`
	Weights string `json:"weights"`
	X       string `json:"x"`
	Y       string `json:"y"`
}

// quantRoot walks up to the go.mod, since tests run from their own directory.
func quantRoot(t *testing.T) string {
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

func loadQuantFixture(t *testing.T, name string) quantFixture {
	t.Helper()
	dir := filepath.Join(quantRoot(t), "testdata", "gemma", "quants")

	raw, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		t.Skipf("quant fixtures missing (%v) — see ref/gemma/README.md", err)
	}
	var index map[string]quantIndexEntry
	if err := json.Unmarshal(raw, &index); err != nil {
		t.Fatal(err)
	}
	entry, ok := index[name]
	if !ok {
		t.Fatalf("fixture %q is not in index.json", name)
	}

	f := quantFixture{Rows: entry.Rows, Cols: entry.Cols}
	if f.Weights, err = os.ReadFile(filepath.Join(dir, entry.Weights)); err != nil {
		t.Fatal(err)
	}
	if entry.X != "" {
		f.X = readFloats(t, filepath.Join(dir, entry.X))
	}
	if entry.Y != "" {
		f.Y = readFloats(t, filepath.Join(dir, entry.Y))
	}
	return f
}

func readFloats(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// compareFloats reports the worst absolute gap relative to the scale of the
// expected values, the same way pockettts/internal/reference does.
func compareFloats(t *testing.T, name string, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	var norm float64
	for _, v := range want {
		norm += float64(v) * float64(v)
	}
	scale := float32(math.Sqrt(norm/float64(len(want)))) + 1e-6

	var worst float32
	var at int
	for i := range got {
		if gap := float32(math.Abs(float64(got[i] - want[i]))); gap > worst {
			worst, at = gap, i
		}
	}
	if relative := worst / scale; relative > tolerance {
		t.Errorf("%s: gap %.3g at index %d (got %.6f, want %.6f), %.3f%% of the scale — beyond %.3f%%",
			name, worst, at, got[at], want[at], relative*100, tolerance*100)
		return
	}
	t.Logf("%s: max gap %.3g (%.4f%% of the scale)", name, worst, worst/scale*100)
}
