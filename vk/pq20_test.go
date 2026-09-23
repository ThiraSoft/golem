package vk

import (
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// aPQ2_0 opens one PQ2_0 matrix out of the Bonsai checkpoint.
func aPQ2_0(tb testing.TB, tensorName string) (*tensors.GGUF, nn.Matrix) {
	tb.Helper()
	path := os.Getenv("GOLEM_MODEL_BONSAI_PQ20")
	if path == "" {
		path = "/mnt/data/LLMs_models/prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("PQ2_0 model not found at %s", path)
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		tb.Fatal(err)
	}
	t, err := g.Get(tensorName)
	if err != nil {
		g.Close()
		tb.Fatalf("%s missing %s: %v", path, tensorName, err)
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok || q != nn.PQ2_0 {
		g.Close()
		tb.Fatalf("%s is not PQ2_0 (got %s)", tensorName, t.DType)
	}
	return g, nn.Matrix{Data: t.Raw, Quant: q, Cols: t.Shape[0], Rows: t.Shape[1]}
}

// TestVulkanPQ20MatVecMatchesCPU is shaders/matvec.comp under -DPQ20 at every
// width the door builds, on real tensors from the Bonsai checkpoint.
func TestVulkanPQ20MatVecMatchesCPU(t *testing.T) {
	d := open(t)
	defer d.Close()

	for _, tensorName := range []string{"blk.0.ffn_down.weight", "blk.0.attn_qkv.weight"} {
		t.Run(tensorName, func(t *testing.T) {
			g, m := aPQ2_0(t, tensorName)
			defer g.Close()

			for _, columns := range []int{1, 2, 4, 8, 16} {
				t.Run(itoa(columns), func(t *testing.T) {
					batch := columnsOf(m.Cols, columns)
					want := kQuantAgainstQ8_0(m, batch, columns)

					q, scales := q80Batch(batch, m.Cols, columns)
					got := quantProductWide(t, d, m, q, scales, columns)

					var worst, scale float64
					var atCol, atRow int
					for c := range got {
						for i := range got[c] {
							scale = math.Max(scale, math.Abs(float64(want[c][i])))
							if gap := math.Abs(float64(got[c][i] - want[c][i])); gap > worst {
								worst, atCol, atRow = gap, c, i
							}
						}
					}
					if worst > 1e-3*scale {
						t.Fatalf("column %d row %d: %v against the processor's %v, %g of a peak of %g",
							atCol, atRow, got[atCol][atRow], want[atCol][atRow], worst, scale)
					}
					t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, columns, worst, scale)
				})
			}
		})
	}
}

// TestVulkanPQ20MatMulMatchesCPU is shaders/matmul_coop.comp under -DPQ20 at every
// tiled width the door builds, on real tensors from the Bonsai checkpoint.
func TestVulkanPQ20MatMulMatchesCPU(t *testing.T) {
	d := open(t)
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("no cooperative matrices on this device")
	}

	for _, tensorName := range []string{"blk.0.ffn_down.weight", "blk.0.attn_qkv.weight"} {
		t.Run(tensorName, func(t *testing.T) {
			g, m := aPQ2_0(t, tensorName)
			defer g.Close()

			for _, columns := range []int{32, 64, 128, 256, 512} {
				t.Run(itoa(columns), func(t *testing.T) {
					batch := columnsOf(m.Cols, columns)
					want := kQuantAgainstQ8_0(m, batch, columns)

					mm, err := NewMatMulQuant(d, m.Data, m.Rows, m.Cols, columns, true, nn.PQ2_0)
					if err != nil {
						t.Fatal(err)
					}
					defer mm.Close()
					for c := 0; c < columns; c++ {
						if err := mm.SetColumn(c, oneColumn(batch, c)); err != nil {
							t.Fatal(err)
						}
					}
					got := make([][]float32, columns)
					for c := range got {
						got[c] = make([]float32, m.Rows)
					}
					if err := mm.Run(got); err != nil {
						t.Fatal(err)
					}

					var worst, scale float64
					var atCol, atRow int
					for c := range got {
						for i := range got[c] {
							scale = math.Max(scale, math.Abs(float64(want[c][i])))
							if gap := math.Abs(float64(got[c][i] - want[c][i])); gap > worst {
								worst, atCol, atRow = gap, c, i
							}
						}
					}
					if worst > 2e-3*scale {
						t.Fatalf("column %d row %d: %v against the processor's %v, %g of a peak of %g",
							atCol, atRow, got[atCol][atRow], want[atCol][atRow], worst, scale)
					}
					t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, columns, worst, scale)
				})
			}
		})
	}
}
