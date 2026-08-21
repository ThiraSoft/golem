package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The packed matrix against the same matrix unpacked, over row ranges that do
// not begin or end on a group and over a matrix whose last rows are not in one.
// The two do not agree to the bit — the packed product sums a block's eight
// rows in lanes and the row-layout one folds each row's own lanes — but they
// agree to rounding, and a row outside the range must not be written at all.
func TestPackedMatrixAgreesWithTheRowLayout(t *testing.T) {
	rng := rand.New(rand.NewSource(29))

	for _, rows := range []int{8, 24, 30, 65} {
		for _, cols := range []int{32, 256, 2080} {
			blocks := cols / QuantBlock
			w := make([]byte, rows*blocks*q4_0BlockBytes)
			for i := range w {
				w[i] = byte(rng.Intn(256))
			}
			for i := 0; i < rows*blocks; i++ {
				w[i*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(2))
			}
			batch := NewBatch(cols, 6)
			for c := 0; c < batch.Size; c++ {
				for i := range batch.F[c] {
					batch.F[c][i] = rng.Float32()*2 - 1
				}
			}
			batch.Quantize()

			plain := Matrix{Data: w, Quant: Q4_0, Rows: rows, Cols: cols}
			packed := plain
			packed.Repack()

			want := make([][]float32, batch.Size)
			for c := range want {
				want[c] = make([]float32, rows)
			}
			plain.MatVecBatch(batch, want)

			for _, span := range [][2]int{{0, rows}, {1, rows - 1}, {3, 11}, {rows - 3, rows}} {
				start, end := span[0], span[1]
				if start < 0 || end > rows || start >= end {
					continue
				}
				got := make([][]float32, batch.Size)
				for c := range got {
					got[c] = make([]float32, rows)
					for r := range got[c] {
						got[c][r] = float32(math.NaN())
					}
				}
				packed.MatVecRows(batch, got, start, end)

				for c := 0; c < batch.Size; c++ {
					for r := 0; r < rows; r++ {
						if r < start || r >= end {
							if !math.IsNaN(float64(got[c][r])) {
								t.Fatalf("rows=%d cols=%d [%d,%d): row %d was written", rows, cols, start, end, r)
							}
							continue
						}
						gap := math.Abs(float64(got[c][r] - want[c][r]))
						if gap > 1e-3*math.Abs(float64(want[c][r]))+1e-4 {
							t.Fatalf("rows=%d cols=%d column %d row %d: packed %.6f, plain %.6f",
								rows, cols, c, r, got[c][r], want[c][r])
						}
					}
				}
			}
		}
	}
}
