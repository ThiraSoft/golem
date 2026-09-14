package nomic

// The card against the same llama.cpp recording the processor is held to, and
// against the processor itself on a batch of texts.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

// bars are what the card is held to. An fp16 file goes up as it is; any other
// is widened to fp16, which is not what ggml computes for that format — it
// quantizes the activation instead — so the bar is that format's own noise:
// llama.cpp's f16 and Q8_0 recordings of the same prompt are 0.99952 apart.
func bars(m *Model) (rel, cos float64) {
	if m.W.Blocks[0].QKV.Quant == nn.F16 {
		return 5e-2, 0.9998
	}
	return 0.15, 0.999
}

func vulkanModel(t *testing.T) *Model {
	m := openModel(t)
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	return m
}

func TestVulkanWaypointsMatchLlamaCpp(t *testing.T) {
	m := vulkanModel(t)
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
	m.trace = nil

	// The card does not round the operand of an fp16 product to fp16 as ggml's
	// processor does, and sums everything in float32 across lanes; its gap to
	// the recording is that, compounded over twelve blocks.
	for name, entry := range rec.Tensors {
		base := name
		if i := strings.LastIndexByte(name, '-'); i > 0 {
			base = name[:i]
		}
		switch base {
		case "kqv_out", "ffn_inp", "ffn_moe_logits", "ffn_moe_weights", "result_embd":
		default:
			continue
		}
		want := readFloats(t, filepath.Join(dir, entry.File))
		have := got[name]
		if len(have) != len(want) {
			t.Errorf("%s: %d values, want %d", name, len(have), len(want))
			continue
		}
		var peak, diff float64
		for i := range want {
			peak = math.Max(peak, math.Abs(float64(want[i])))
			diff = math.Max(diff, math.Abs(float64(have[i]-want[i])))
		}
		t.Logf("%-20s rel %.3g", name, diff/peak)
		if limit, _ := bars(m); diff/peak > limit {
			t.Errorf("%s: relative gap %.3g", name, diff/peak)
		}
	}
	want := readFloats(t, filepath.Join(dir, "embedding.bin"))
	cos := cosine(vecs[0], want)
	t.Logf("cosine with llama.cpp's embedding: %.9f", cos)
	if _, bar := bars(m); cos < bar {
		t.Errorf("cosine %.9f below %g", cos, bar)
	}
}

// A batch of texts on the card against the same texts on the processor, and
// each text alone on the card against the same text in the batch — which has
// to be the same float, because every tile of the card's product and every
// row of its attention depends on its own row and column alone.
func TestVulkanAgreesWithTheProcessor(t *testing.T) {
	m := openModel(t)
	var texts [][]int32
	for i := 0; i < 12; i++ {
		texts = append(texts, m.Tokenize(strings.Repeat("search_document: golem reads a text once ", 1+i%5)+string(rune('a'+i))))
	}
	cpu, err := m.Embed(texts)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.UseVulkan(); err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	gpu, err := m.Embed(texts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range texts {
		if _, bar := bars(m); cosine(cpu[i], gpu[i]) < bar {
			c := cosine(cpu[i], gpu[i])
			t.Errorf("text %d: the card and the processor are %.7f apart", i, c)
		}
		alone, err := m.Embed(texts[i : i+1])
		if err != nil {
			t.Fatal(err)
		}
		for j := range alone[0] {
			if alone[0][j] != gpu[i][j] {
				t.Fatalf("text %d alone differs from the same text in a batch at %d: %g against %g",
					i, j, alone[0][j], gpu[i][j])
			}
		}
	}
}
