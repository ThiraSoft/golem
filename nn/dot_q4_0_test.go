package nn

import "testing"

// TestMatVecQ4_0 checks the product against the one ggml computed on the same
// weights and the same activation. Both quantize the activation to Q8_0 and
// multiply in integers, so the two results should agree to rounding.
func TestMatVecQ4_0(t *testing.T) {
	f := loadQuantFixture(t, "q4_0_matvec")

	x := NewBatch(f.Cols, 1)
	x.Set(0, f.X)

	y := [][]float32{make([]float32, f.Rows)}
	MatVecQ4_0(f.Weights, x, f.Rows, f.Cols, y)

	compareFloats(t, "q4_0 matvec", y[0], f.Y, 1e-5)
}
