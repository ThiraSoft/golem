package nn

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type mropeIndex struct {
	HeadSize int     `json:"head_size"`
	NDims    int     `json:"n_dims"`
	NHead    int     `json:"n_head"`
	Base     float64 `json:"base"`
	Sections [4]int  `json:"sections"`
	Cases    []struct {
		Name string `json:"name"`
		T    int    `json:"t"`
		H    int    `json:"h"`
		W    int    `json:"w"`
		E    int    `json:"e"`
	} `json:"cases"`
}

// readFloats is quants_reference_test.go's, which reads the same little-endian
// float32 files every recorder in ref/ writes.

// The rotation against ggml's own, over positions text alone could never
// produce. What this pins is the round robin and the frequency that does not
// reset with it — the two things a transcription gets wrong without saying so.
func TestMRoPEMatchesGGML(t *testing.T) {
	dir := filepath.Join("..", "testdata", "qwen35", "mrope")
	raw, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		t.Skipf("fixtures missing (%v) — see ref/README.md", err)
	}
	var idx mropeIndex
	if err := json.Unmarshal(raw, &idx); err != nil {
		t.Fatal(err)
	}

	for _, c := range idx.Cases {
		t.Run(c.Name, func(t *testing.T) {
			x := readFloats(t, filepath.Join(dir, c.Name+".x.bin"))
			want := readFloats(t, filepath.Join(dir, c.Name+".y.bin"))

			var table RoPETable
			table.PrepareMulti(idx.NDims, [4]int{c.T, c.H, c.W, c.E}, idx.Base, Sections(idx.Sections), nil)

			got := append([]float32(nil), x...)
			for h := 0; h < idx.NHead; h++ {
				head := got[h*idx.HeadSize : (h+1)*idx.HeadSize]
				table.Apply(head[:idx.NDims])
			}

			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
					t.Fatalf("element %d: %v against ggml's %v (%v away)", i, got[i], want[i], diff)
				}
			}

			// A rotation that ignored h and w would still pass every
			// degenerate case, so the spread ones have to be shown to be
			// spread: taking t alone must land somewhere else. Without this,
			// a fixture set recorded with equal components would look like
			// coverage it is not.
			if c.T == c.H && c.T == c.W {
				return
			}
			var scalar RoPETable
			scalar.Prepare(idx.NDims, c.T, idx.Base, nil)
			naive := append([]float32(nil), x...)
			for h := 0; h < idx.NHead; h++ {
				scalar.Apply(naive[h*idx.HeadSize:][:idx.NDims])
			}
			same := true
			for i := range want {
				if math.Abs(float64(naive[i]-want[i])) > 1e-6 {
					same = false
					break
				}
			}
			if same {
				t.Fatal("the scalar rotation matches this case too, so it pins nothing about the sections")
			}
		})
	}
}
