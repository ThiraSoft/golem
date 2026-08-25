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

// quantDirs are the recordings, one per model that was asked for one. A format
// is recorded where a model that has it lives: Gemma 4 gives Q4_0 and Q6_K,
// and Qwen3.8-27B is where the Q4_1 that its ffn_down is stored in comes from.
var quantDirs = []string{
	filepath.Join("testdata", "gemma", "quants"),
	filepath.Join("testdata", "qwen", "quants"),
}

func loadQuantFixture(t *testing.T, name string) quantFixture {
	t.Helper()
	root := quantRoot(t)

	var dir string
	var entry quantIndexEntry
	var found, any bool
	for _, candidate := range quantDirs {
		at := filepath.Join(root, candidate)
		raw, err := os.ReadFile(filepath.Join(at, "index.json"))
		if err != nil {
			continue
		}
		any = true
		var index map[string]quantIndexEntry
		if err := json.Unmarshal(raw, &index); err != nil {
			t.Fatal(err)
		}
		if e, ok := index[name]; ok {
			dir, entry, found = at, e, true
			break
		}
	}
	if !any {
		t.Skip("quant fixtures missing — see ref/gemma/README.md")
	}
	if !found {
		t.Skipf("fixture %q is in none of the recordings — see ref/README.md", name)
	}
	var err error

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
