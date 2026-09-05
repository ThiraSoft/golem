package qwen35

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ThiraSoft/golem/imageio"
	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/tensors"
)

type visionIndex struct {
	Image        string `json:"image"`
	NImageTokens int    `json:"n_image_tokens"`
	Tensors      map[string]struct {
		File string `json:"file"`
		Ne   []int  `json:"ne"`
	} `json:"tensors"`
}

func visionFixture(t *testing.T) (visionIndex, string) {
	t.Helper()
	dir := filepath.Join("..", "testdata", "qwen35", "vision")
	raw, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		t.Skipf("fixtures missing (%v) — see ref/README.md", err)
	}
	var idx visionIndex
	if err := json.Unmarshal(raw, &idx); err != nil {
		t.Fatal(err)
	}
	return idx, dir
}

func readFixtureFloats(t *testing.T, path string) []float32 {
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

// The worst gap relative to the scale of what was expected, which is how the
// other reference tests in this repository measure.
func worstGap(got, want []float32) (float64, int) {
	var scale float64
	for _, v := range want {
		if d := math.Abs(float64(v)); d > scale {
			scale = d
		}
	}
	if scale == 0 {
		scale = 1
	}
	worst, at := 0.0, -1
	for i := range want {
		if d := math.Abs(float64(got[i]-want[i])) / scale; d > worst {
			worst, at = d, i
		}
	}
	return worst, at
}

// The tower against llama.cpp, waypoint by waypoint.
//
// The two before any block come first on purpose: if patch_bias differs the
// convolutions or the patch order are wrong, and if only inp_pos_emb does it
// is the interpolation or the reordering of the learned table. Only then do
// the blocks mean anything.
func TestVisionTowerMatchesLlamaCpp(t *testing.T) {
	heavy.Skip(t, "it runs a checkpoint of tens of gigabytes")
	idx, dir := visionFixture(t)

	g, err := tensors.OpenGGUF(qwen38mmproj)
	if err != nil {
		t.Skipf("projector: %v", err)
	}
	defer g.Close()
	cfg, err := LoadVisionConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := LoadVisionWeights(g, cfg)
	if err != nil {
		t.Fatal(err)
	}
	tower := NewVisionTower(cfg, w)
	tower.Trace()

	f, err := os.Open(filepath.Join("..", idx.Image))
	if err != nil {
		t.Skipf("image: %v", err)
	}
	defer f.Close()
	im, err := imageio.Decode(f)
	if err != nil {
		t.Fatal(err)
	}

	rows := tower.Encode(im)
	if len(rows) != idx.NImageTokens {
		t.Fatalf("%d rows, and llama.cpp made %d — the grid is not the same, so nothing below compares",
			len(rows), idx.NImageTokens)
	}

	const tolerance = 0.02
	for _, name := range []string{"patch_bias", "inp_pos_emb", "layer_out-0", "layer_out-1", "layer_out-13", "layer_out-26"} {
		meta, ok := idx.Tensors[name]
		if !ok {
			t.Logf("%s: not recorded, skipped", name)
			continue
		}
		got := tower.Waypoint(name)
		if got == nil {
			t.Errorf("%s: not traced", name)
			continue
		}
		want := readFixtureFloats(t, filepath.Join(dir, meta.File))
		if len(got) != len(want) {
			t.Errorf("%s: %d values against %d %v", name, len(got), len(want), meta.Ne)
			continue
		}
		gap, at := worstGap(got, want)
		if gap > tolerance {
			t.Errorf("%s: worst gap %.4f of the scale, at element %d (%v against %v)",
				name, gap, at, got[at], want[at])
			continue
		}
		t.Logf("%s: worst gap %.5f", name, gap)
	}
}
