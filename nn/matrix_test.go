package nn

import "testing"

// TestMatrixQ4_0 routes the same fixture through Matrix rather than through
// MatVecQ4_0 directly.
func TestMatrixQ4_0(t *testing.T) {
	f := loadQuantFixture(t, "q4_0_matvec")

	m := Matrix{Data: f.Weights, Quant: Q4_0, Rows: f.Rows, Cols: f.Cols}
	x := NewBatch(f.Cols, 1)
	x.Set(0, f.X)

	y := make([]float32, f.Rows)
	m.MatVec(x, y)

	compareFloats(t, "matrix q4_0", y, f.Y, 1e-5)
}

// TestMatrixBias adds a bias and checks it lands.
func TestMatrixBias(t *testing.T) {
	f := loadQuantFixture(t, "q4_0_matvec")

	bias := make([]float32, f.Rows)
	for i := range bias {
		bias[i] = float32(i) * 0.5
	}
	m := Matrix{Data: f.Weights, Quant: Q4_0, Rows: f.Rows, Cols: f.Cols, Bias: bias}
	x := NewBatch(f.Cols, 1)
	x.Set(0, f.X)

	y := make([]float32, f.Rows)
	m.MatVec(x, y)

	want := make([]float32, f.Rows)
	for i := range want {
		want[i] = f.Y[i] + bias[i]
	}
	compareFloats(t, "matrix q4_0 with bias", y, want, 1e-5)
}

// TestMatrixRowQ6_K reads one embedding row.
func TestMatrixRowQ6_K(t *testing.T) {
	f := loadQuantFixture(t, "q6_k_dequant")

	m := Matrix{Data: f.Weights, Quant: Q6_K, Rows: f.Rows, Cols: f.Cols}
	got := make([]float32, f.Cols)
	m.Row(2, got)

	compareFloats(t, "matrix q6_k row 2", got, f.Y[2*f.Cols:3*f.Cols], 1e-6)
}

// TestMatrixRowQ4_0 checks the Row path on Q4_0 as well. The fixture exists
// and nothing else reads it: a dequantized row is what an embedding table
// stored in Q4_0 would give, and the nibble layout is worth confirming once
// away from the product, where a mistake could still cancel out.
func TestMatrixRowQ4_0(t *testing.T) {
	f := loadQuantFixture(t, "q4_0_dequant")

	m := Matrix{Data: f.Weights, Quant: Q4_0, Rows: f.Rows, Cols: f.Cols}
	got := make([]float32, f.Rows*f.Cols)
	for r := 0; r < f.Rows; r++ {
		m.Row(r, got[r*f.Cols:(r+1)*f.Cols])
	}

	compareFloats(t, "matrix q4_0 rows", got, f.Y, 1e-6)
}
