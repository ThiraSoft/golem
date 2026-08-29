package nn

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The tower's rotation against ggml's. What it pins is the one thing that
// separates it from the trunk's: the frequency starts again at every section,
// so the first pair of the column section turns as fast as the first pair of
// the row section rather than eighteen steps slower.
func TestVisionRoPEMatchesGGML(t *testing.T) {
	dir := filepath.Join("..", "testdata", "qwen35", "vrope")
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
			// The whole head rotates here, not idx.NDims of it: for VISION
			// ggml passes n_dims as the pairing offset and rotates ne0.
			table.PrepareVision(idx.HeadSize, [2]int{c.T, c.H}, idx.Base, Sections(idx.Sections), nil)

			got := append([]float32(nil), x...)
			for h := 0; h < idx.NHead; h++ {
				table.Apply(got[h*idx.HeadSize : (h+1)*idx.HeadSize])
			}
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-6 {
					t.Fatalf("element %d: %v against ggml's %v (%v away)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

// The two rotations must not agree, or one of them is not being tested. With
// the same positions and the same sections, a resetting frequency and a
// running one give different angles from the second section on.
func TestVisionAndTrunkRotationsDiffer(t *testing.T) {
	// Sections narrow enough that both rules accept them: the trunk wants them
	// to sum to no more than half the head, the tower sizes them off the whole
	// head. Six each is the widest that fits both at these dimensions.
	sections := Sections{6, 6, 6, 0}
	var vision, trunk RoPETable
	vision.PrepareVision(36, [2]int{7, 11}, 10000, sections, nil)
	trunk.PrepareMulti(36, [4]int{7, 11, 7, 11}, 10000, sections, nil)
	for i := range vision.Cos {
		if vision.Cos[i] != trunk.Cos[i] {
			return
		}
	}
	t.Fatal("the vision rotation equals the trunk's, so one of them is wrong")
}

// The two rules share a table type and therefore a cache key. A table prepared
// one way and then the other must recompute, not answer from the first.
func TestTheTwoRulesDoNotShareACachedTable(t *testing.T) {
	sections := Sections{6, 6, 6, 0}
	var table RoPETable

	table.PrepareVision(36, [2]int{7, 11}, 10000, sections, nil)
	vision := append([]float32(nil), table.Cos...)

	table.PrepareMulti(36, [4]int{7, 11, 7, 11}, 10000, sections, nil)
	trunk := append([]float32(nil), table.Cos...)

	table.PrepareVision(36, [2]int{7, 11}, 10000, sections, nil)
	again := table.Cos

	same := true
	for i := range vision {
		if vision[i] != trunk[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("the trunk's rotation came back unchanged from the vision one")
	}
	for i := range vision {
		if vision[i] != again[i] {
			t.Fatalf("entry %d: the vision table did not come back, %v against %v", i, again[i], vision[i])
		}
	}
}
