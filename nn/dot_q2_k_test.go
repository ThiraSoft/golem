package nn

import (
	"math"
	"testing"
)

// TestDequantizeQ2_KMatchesGGML holds golem's reader to ggml's own, on a real
// tensor, weight for weight.
//
// The fixture is ref/dump_quants.cpp's, recorded from a Qwen3-4B-Q2_K where
// blk.0.attn_q.weight is the two-bit tier. Weight for weight and not nearly:
// nothing here rounds, so the two dequantisers compute the same float32 from
// the same bytes or one of them is walking the superblock wrongly.
//
// That walk is the whole risk. Q2_K packs four blocks of thirty-two into one
// window of qs at four bit positions, and its sixteen scale bytes each carry a
// scale and a minimum in two nibbles — so a reader that took the groups in the
// wrong order, or the nibbles the wrong way round, would still produce
// plausible weights on every row.
func TestDequantizeQ2_KMatchesGGML(t *testing.T) {
	f := loadQuantFixture(t, "q2_k_dequant")
	got := make([]float32, f.Rows*f.Cols)
	stride := f.Cols / SuperBlock * q2_kBlockBytes
	for r := 0; r < f.Rows; r++ {
		DequantizeQ2_K(f.Weights[r*stride:(r+1)*stride], f.Cols, got[r*f.Cols:])
	}
	for i := range got {
		if got[i] != f.Y[i] {
			t.Fatalf("weight %d (row %d, column %d): %v, ggml says %v",
				i, i/f.Cols, i%f.Cols, got[i], f.Y[i])
		}
	}
}

// And the product ggml itself performed on those weights, which is the join
// the dequantiser cannot make: it checks that the integer kernel adds the same
// terms in the same groups, minima included.
//
// The tolerance is on the row's own energy — the sum of the magnitudes it adds
// up — because a dot of a few thousand terms cancels down to a fraction of
// them, and the activation's eight-bit rounding does not cancel with it. At two
// bits a weight the cancellation is severe, which is why this is measured
// against the terms and not against the answer.
func TestMatVecQ2_KMatchesGGML(t *testing.T) {
	f := loadQuantFixture(t, "q2_k_matvec")

	x := NewBatch(f.Cols, 1)
	copy(x.F[0], f.X)
	x.QuantizeK()

	got := [][]float32{make([]float32, f.Rows)}
	MatVecQ2_K(f.Weights, x, f.Rows, f.Cols, got)

	stride := f.Cols / SuperBlock * q2_kBlockBytes
	row := make([]float32, f.Cols)
	for r := 0; r < f.Rows; r++ {
		DequantizeQ2_K(f.Weights[r*stride:(r+1)*stride], f.Cols, row)
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
