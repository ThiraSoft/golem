package nn

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type interpIndex struct {
	SrcW     int `json:"src_w"`
	SrcH     int `json:"src_h"`
	Channels int `json:"channels"`
	Cases    []struct {
		Name string `json:"name"`
		DstW int    `json:"dst_w"`
		DstH int    `json:"dst_h"`
	} `json:"cases"`
}

// The resize against ggml's own, over the scales a dynamic patch grid asks
// for. What it pins is the antialias: the support widens as the grid shrinks,
// and the weights are divided by what was gathered rather than by their
// nominal sum.
func TestInterpolateMatchesGGML(t *testing.T) {
	dir := filepath.Join("..", "testdata", "qwen35", "interp")
	raw, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		t.Skipf("fixtures missing (%v) — see ref/README.md", err)
	}
	var idx interpIndex
	if err := json.Unmarshal(raw, &idx); err != nil {
		t.Fatal(err)
	}

	for _, c := range idx.Cases {
		t.Run(c.Name, func(t *testing.T) {
			x := readFloats(t, filepath.Join(dir, c.Name+".x.bin"))
			want := readFloats(t, filepath.Join(dir, c.Name+".y.bin"))

			got := ResizeBilinearAntialias(x, idx.SrcW, idx.SrcH, c.DstW, c.DstH, idx.Channels)
			if len(got) != len(want) {
				t.Fatalf("%d values, wanted %d", len(got), len(want))
			}
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
					t.Fatalf("element %d: %v against ggml's %v (%v away)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

// A flat field must stay flat, at every scale. The weights are normalized by
// what was actually gathered, so an output whose support falls off the edge is
// still an average and not a darkened one — which is exactly what an
// unnormalized bilinear gets wrong, and what no picture shows.
func TestFlatFieldSurvivesEveryScale(t *testing.T) {
	src := make([]float32, 48*48)
	for i := range src {
		src[i] = 0.375
	}
	for _, to := range [][2]int{{48, 48}, {32, 32}, {64, 16}, {64, 64}, {37, 23}} {
		got := ResizeBilinearAntialias(src, 48, 48, to[0], to[1], 1)
		if len(got) != to[0]*to[1] {
			t.Fatalf("%dx%d: %d values", to[0], to[1], len(got))
		}
		for i, v := range got {
			if math.Abs(float64(v-0.375)) > 1e-6 {
				t.Fatalf("%dx%d, element %d: %v, wanted 0.375", to[0], to[1], i, v)
			}
		}
	}
}

// Shrinking must actually average rather than pick. A grid that alternates
// between two values has a mean an ordinary sampler would miss entirely: it
// would land on one value or the other, never between them.
func TestShrinkingAveragesRatherThanSamples(t *testing.T) {
	src := make([]float32, 48*48)
	for y := 0; y < 48; y++ {
		for x := 0; x < 48; x++ {
			if (x+y)%2 == 0 {
				src[y*48+x] = 1
			}
		}
	}
	got := ResizeBilinearAntialias(src, 48, 48, 12, 12, 1)
	for i, v := range got {
		if v < 0.2 || v > 0.8 {
			t.Fatalf("element %d: %v — a checkerboard shrunk fourfold should land near its mean", i, v)
		}
	}
}
