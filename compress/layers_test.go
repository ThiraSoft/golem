package compress

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

// Whether one block's weights are worth anything to the next.
//
// The idea is old and keeps coming back: adjacent transformer blocks are said
// to learn nearly the same transformation, so store one and give the next a
// delta, which quantizes to fewer bits because it is smaller. Everything rests
// on one number — how much smaller the delta is than the matrix — and that
// number is cheap to look at.
//
// The reference to beat is √2. Two matrices of equal norm and no relation have
// a difference of √2 times either of them, so a ratio at or above √2 means the
// blocks share nothing at all and a delta costs more than the matrix.
//
// GOLEM_MODEL_QWEN points at the checkpoint; the test says nothing without it.
func TestAdjacentLayersShareNothing(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL_QWEN")
	if path == "" {
		t.Skip("GOLEM_MODEL_QWEN unset")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Skip(err)
	}
	defer g.Close()
	for _, mat := range []string{"attn_q", "ffn_down", "ffn_gate"} {
		worst, at := math.Inf(1), 0
		for k := 1; k < 28; k++ {
			a, ok1 := g.Tensors[fmt.Sprintf("blk.%d.%s.weight", k-1, mat)]
			b, ok2 := g.Tensors[fmt.Sprintf("blk.%d.%s.weight", k, mat)]
			if !ok1 || !ok2 {
				break
			}
			wa, err := a.F32()
			if err != nil {
				t.Fatal(err)
			}
			wb, err := b.F32()
			if err != nil {
				t.Fatal(err)
			}
			var num, den float64
			for i := range wa {
				d := float64(wb[i] - wa[i])
				num += d * d
				den += float64(wb[i]) * float64(wb[i])
			}
			if r := math.Sqrt(num / den); r < worst {
				worst, at = r, k
			}
		}
		t.Logf("%-9s the closest pair of blocks differs by %.3f of a matrix, at block %d (√2 = 1.414 means unrelated)",
			mat, worst, at)
		if worst < 1.2 {
			t.Errorf("%s: blocks %d and %d are close enough that a delta might pay — worth looking at", mat, at-1, at)
		}
	}
}
