package qwen35

import (
	"testing"

	"github.com/ThiraSoft/golem/internal/heavy"
)

// What the card does with a position that has three axes, in one model load.
//
// Two assertions, and neither is enough alone. The first says the rotation
// collapses exactly when the axes agree, which is what text depends on. The
// second says the axes reach the shader at all — without it the first passes
// just as well on a pipeline that was never wired, because a shader given no
// sections takes the scalar path and answers text correctly.
//
// They share a model because loading one costs 12.8 GiB of a 16 GiB card, and
// qwen35's suite already spends nine of those.
func TestVulkanMRoPE(t *testing.T) {
	heavy.Skip(t, "it takes forty seconds, the longest check in the package")
	m, err := Open(qwen38, 512)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer m.Close()
	if err := m.UseVulkan(); err != nil {
		t.Skipf("vulkan: %v", err)
	}

	tokens := []int32{9707, 11, 1879, 0, 358, 1079, 264, 1273}

	t.Run("text is unmoved", func(t *testing.T) {
		byRun := m.ForwardBatch(tokens, 0)
		held := make([][]float32, len(byRun))
		for i := range byRun {
			held[i] = append([]float32(nil), byRun[i]...)
		}

		m.Reset()
		at := make([]Place, len(tokens))
		for i := range at {
			at[i] = Place{Slot: 0, Pos: i, T: i, H: i, W: i}
		}
		byPlace := m.ForwardPlaces(tokens, at)

		for c := range held {
			for i := range held[c] {
				if held[c][i] != byPlace[c][i] {
					t.Fatalf("token %d, element %d: %v against %v", c, i, held[c][i], byPlace[c][i])
				}
			}
		}
	})

	t.Run("the axes reach the shader", func(t *testing.T) {
		if m.Cfg.RoPESections[0] == 0 {
			t.Skip("this checkpoint declares no sections")
		}
		short := tokens[:4]

		m.Reset()
		plain := make([]Place, len(short))
		for i := range plain {
			plain[i] = Place{Slot: 0, Pos: i, T: i, H: i, W: i}
		}
		first := m.ForwardPlaces(short, plain)
		held := append([]float32(nil), first[len(first)-1]...)

		// The same cache indices, one axis moved. The keys land in the same
		// entries; only the rotation may differ.
		m.Reset()
		spread := make([]Place, len(short))
		for i := range spread {
			spread[i] = Place{Slot: 0, Pos: i, T: i, H: i + 7, W: i}
		}
		second := m.ForwardPlaces(short, spread)

		for i := range held {
			if held[i] != second[len(second)-1][i] {
				return
			}
		}
		t.Fatal("moving H changed nothing: the sections are not reaching the shader")
	})
}
