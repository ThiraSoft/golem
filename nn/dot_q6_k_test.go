package nn

import (
	"math"
	"math/rand"
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

// The kernel and the portable form compute the same integers in the same
// order, so they agree bit for bit.
func TestDotQ6_KAVX2MatchesTheGoForm(t *testing.T) {
	if !avx2 {
		t.Skip("no AVX2")
	}
	r := rand.New(rand.NewSource(7))
	for _, superblocks := range []int{1, 2, 6} {
		n := superblocks * SuperBlock
		w := make([]byte, superblocks*q6_kBlockBytes)
		for i := range w {
			w[i] = byte(r.Intn(256))
		}
		x := NewBatch(n, 1)
		for i := range x.F[0] {
			x.F[0][i] = r.Float32()*2 - 1
		}
		x.QuantizeK()

		want := dotQ6_KGo(w, x.QK, x.BSums, x.KScales, n)
		got := dotQ6_KAVX2(&w[0], &x.QK[0], &x.BSums[0], &x.KScales[0], n)
		if got != want {
			t.Fatalf("%d superblocks: %v against %v", superblocks, got, want)
		}
	}
}
