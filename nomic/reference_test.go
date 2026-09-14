package nomic

// Parity with llama.cpp, waypoint by waypoint: ref/nomic/short.run through
// build/ref/dump_layers, on the CPU backend with flash attention off. The
// fixture for an f16 checkpoint is testdata/nomic/layers and for a Q8_0 one
// testdata/nomic/layers_q8; the test picks by what the file holds.
//
// Every waypoint is compared, in the order the graph produces them, so that
// the first one out of tolerance says where a mistake is rather than the last.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

type recording struct {
	Tokens  []int32 `json:"tokens"`
	Tensors map[string]struct {
		File string   `json:"file"`
		Ne   [4]int64 `json:"ne"`
	} `json:"tensors"`
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
	path := os.Getenv("GOLEM_MODEL_NOMIC")
	if path == "" {
		t.Skip("GOLEM_MODEL_NOMIC is not set")
	}
	m, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func fixtureDir(t *testing.T, m *Model) string {
	sub := "layers"
	switch q := m.W.Blocks[0].QKV.Quant; q {
	case nn.F16:
	case nn.Q8_0:
		sub = "layers_q8"
	default:
		t.Skipf("no fixture recorded for a %s checkpoint", q)
	}
	return filepath.Join(repoRoot(t), "testdata", "nomic", sub)
}

func readFloats(t *testing.T, path string) []float32 {
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

// rank orders waypoints the way the graph computes them.
func rank(name string) (int, int) {
	order := []string{"inp_embd", "inp_norm", "Qcur", "Kcur", "Vcur", "kqv_out", "ffn_inp",
		"ffn_moe_logits", "ffn_moe_weights", "ffn_out", "ffn_moe_out", "result_embd", "embedding"}
	base, block := name, -1
	if i := strings.LastIndexByte(name, '-'); i > 0 {
		if b, err := strconv.Atoi(name[i+1:]); err == nil {
			base, block = name[:i], b
		}
	}
	for k, o := range order {
		if o == base {
			if block < 0 {
				if k >= len(order)-2 {
					return 1 << 20, k
				}
				return -1, k
			}
			return block, k
		}
	}
	return 1 << 21, 0
}

func TestWaypointsMatchLlamaCpp(t *testing.T) {
	m := openModel(t)
	dir := fixtureDir(t, m)
	raw, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		t.Skip("the layer fixtures are not on this machine")
	}
	var rec recording
	if err := json.Unmarshal(raw, &rec); err != nil {
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
	vecs, err := m.Embed([][]int32{rec.Tokens})
	if err != nil {
		t.Fatal(err)
	}
	got["embedding"] = vecs[0]

	names := make([]string, 0, len(rec.Tensors))
	for name := range rec.Tensors {
		names = append(names, name)
	}
	sort.Slice(names, func(a, b int) bool {
		ba, ka := rank(names[a])
		bb, kb := rank(names[b])
		if ba != bb {
			return ba < bb
		}
		return ka < kb
	})

	// Relative to the largest entry of the reference. The engine rounds what
	// ggml rounds and sums in a different order, so the gap is summation
	// order, compounded over twelve blocks of norms.
	//
	// A Q8_0 checkpoint quantizes every activation to eight bits on its way
	// into a product, and a difference of one ulp is enough to move one of
	// them by a whole step. Blocks 0 and 1 are bit for bit what ggml writes,
	// because the norm is ggml's to the bit. From block 2 the gap grows to a
	// couple of per cent: the router's softmax, whose exponential ggml takes
	// from an AVX2 polynomial that is not expf, hands the next norm an ulp.
	// llama.cpp's own f16 and Q8_0 recordings of the same prompt are 0.99952
	// apart at the pooled vector, which is the size of the quantization's own
	// noise; the bar for the cosine is four times tighter than that.
	tolerance, quantized := 2e-3, m.W.Blocks[0].QKV.Quant == nn.Q8_0
	minCosine := 0.99999
	if quantized {
		minCosine = 0.9998
	}
	for _, name := range names {
		if b, _ := rank(name); quantized && b > 1 {
			tolerance = 5e-2
		}
		want := readFloats(t, filepath.Join(dir, rec.Tensors[name].File))
		have, ok := got[name]
		if !ok {
			t.Errorf("%s: the engine never produced it", name)
			continue
		}
		if len(have) != len(want) {
			t.Errorf("%s: %d values, want %d", name, len(have), len(want))
			continue
		}
		var peak, diff float64
		for i := range want {
			peak = math.Max(peak, math.Abs(float64(want[i])))
			diff = math.Max(diff, math.Abs(float64(have[i]-want[i])))
		}
		rel := diff / math.Max(peak, 1e-30)
		t.Logf("%-20s max|Δ| %.3g  peak %.3g  rel %.3g", name, diff, peak, rel)
		if rel > tolerance {
			t.Errorf("%s: relative gap %.3g above %.3g", name, rel, tolerance)
		}
	}

	var dot, na, nb float64
	want := readFloats(t, filepath.Join(dir, "embedding.bin"))
	for i := range want {
		dot += float64(want[i]) * float64(vecs[0][i])
		na += float64(want[i]) * float64(want[i])
		nb += float64(vecs[0][i]) * float64(vecs[0][i])
	}
	cos := dot / math.Sqrt(na*nb)
	t.Logf("cosine with llama.cpp's embedding: %.9f", cos)
	if cos < minCosine {
		t.Errorf("cosine %.9f below %g", cos, minCosine)
	}
}
