package nn

import (
	"math"
	"testing"
)

// The same two joins for Q3_K, which had none: its reader was pinned by a
// hand-built superblock and its kernels by each other, and neither of those is
// ggml. The fixture comes from the same Qwen3-4B-Q2_K, whose output projection
// llama.cpp's two-bit mix stores in the three-bit tier.
func TestDequantizeQ3_KMatchesGGML(t *testing.T) {
	f := loadQuantFixture(t, "q3_k_dequant")
	got := make([]float32, f.Rows*f.Cols)
	stride := f.Cols / SuperBlock * q3_kBlockBytes
	for r := 0; r < f.Rows; r++ {
		DequantizeQ3_K(f.Weights[r*stride:(r+1)*stride], f.Cols, got[r*f.Cols:])
	}
	for i := range got {
		if got[i] != f.Y[i] {
			t.Fatalf("weight %d (row %d, column %d): %v, ggml says %v",
				i, i/f.Cols, i%f.Cols, got[i], f.Y[i])
		}
	}
}

func TestMatVecQ3_KMatchesGGML(t *testing.T) {
	f := loadQuantFixture(t, "q3_k_matvec")

	x := NewBatch(f.Cols, 1)
	copy(x.F[0], f.X)
	x.QuantizeK()

	got := [][]float32{make([]float32, f.Rows)}
	MatVecQ3_K(f.Weights, x, f.Rows, f.Cols, got)

	stride := f.Cols / SuperBlock * q3_kBlockBytes
	row := make([]float32, f.Cols)
	for r := 0; r < f.Rows; r++ {
		DequantizeQ3_K(f.Weights[r*stride:(r+1)*stride], f.Cols, row)
		var energy float64
		for c := range row {
			energy += math.Abs(float64(row[c] * f.X[c]))
		}
		if d := math.Abs(float64(got[0][r] - f.Y[r])); d > 5e-3*energy {
			t.Fatalf("row %d: %v, ggml says %v, gap %g against an energy of %g",
				r, got[0][r], f.Y[r], d, energy)
		}
	}
}
