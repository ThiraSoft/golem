package nn

import "testing"

// TestDequantizeQ6_K checks four real embedding rows against what ggml's own
// to_float produced for the same bytes.
func TestDequantizeQ6_K(t *testing.T) {
	f := loadQuantFixture(t, "q6_k_dequant")

	stride := f.Cols / SuperBlock * q6_kBlockBytes
	got := make([]float32, f.Rows*f.Cols)
	for r := 0; r < f.Rows; r++ {
		DequantizeQ6_K(f.Weights[r*stride:(r+1)*stride], f.Cols, got[r*f.Cols:(r+1)*f.Cols])
	}

	compareFloats(t, "q6_k dequant", got, f.Y, 1e-6)
}
