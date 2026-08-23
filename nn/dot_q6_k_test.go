package nn

import (
	"math"
	"testing"
)

// The integer product has to agree with dequantizing the rows and multiplying
// by hand, which is what DequantizeQ6_K is already pinned against ggml for.
// The activation is quantized to eight bits on the way in, so the agreement is
// relative, not exact.
func TestMatVecQ6_KAgreesWithDequantization(t *testing.T) {
	fx := loadQuantFixture(t, "q6_k_dequant")
	rows, cols := fx.Rows, fx.Cols

	x := NewBatch(cols, 1)
	for i := range x.F[0] {
		x.F[0][i] = float32(math.Sin(float64(i) * 0.37))
	}
	x.QuantizeK()

	// The bound is on the row's own scale — the sum of the magnitudes it adds
	// up — because a dot product of a thousand terms cancels down to a fraction
	// of them, and an eight-bit activation's error does not cancel with it.
	want := make([]float32, rows)
	energy := make([]float64, rows)
	dequantized := fx.Y // ggml's own dequantization of the same rows
	for r := 0; r < rows; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			term := dequantized[r*cols+c] * x.F[0][c]
			sum += term
			energy[r] += math.Abs(float64(term))
		}
		want[r] = sum
	}

	got := [][]float32{make([]float32, rows)}
	MatVecQ6_K(fx.Weights, x, rows, cols, got)

	for r := range got[0] {
		if d := math.Abs(float64(got[0][r] - want[r])); d > 5e-3*energy[r] {
			t.Fatalf("row %d: %v against %v, gap %g", r, got[0][r], want[r], d)
		}
	}
}
