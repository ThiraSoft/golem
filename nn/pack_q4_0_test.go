package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The packed layout carries the same weights: eight rows through the packed
// product, against the same eight rows read from the file's own layout.
func TestPackQ4_0MatchesTheRowLayout(t *testing.T) {
	rng := rand.New(rand.NewSource(19))

	for _, cols := range []int{32, 64, 256, 1536} {
		const rows = PackedRows
		blocks := cols / QuantBlock
		w := make([]byte, rows*blocks*q4_0BlockBytes)
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		for i := 0; i < rows*blocks; i++ {
			w[i*q4_0BlockBytes] = byte(rng.Intn(256))
			w[i*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
		}
		packed := make([]byte, PackedQ4_0Bytes(rows, cols))
		PackQ4_0(w, rows, cols, packed)

		batch := NewBatch(cols, 3)
		for c := 0; c < batch.Size; c++ {
			for i := range batch.F[c] {
				batch.F[c][i] = rng.Float32()*2 - 1
			}
		}
		batch.Quantize()

		for c := 0; c < batch.Size; c++ {
			state := make([]float32, PackedRows)
			dotPackedQ4_0Go(packed, batch, 0, c, cols, state)
			for r := 0; r < rows; r++ {
				var lanes [9]float32
				dotQ4_0Go(w[r*blocks*q4_0BlockBytes:], batch, 0, c, cols, lanes[:])
				want := fold(lanes[:], lanes[8])
				if gap := math.Abs(float64(state[r] - want)); gap > 1e-3*math.Abs(float64(want))+1e-5 {
					t.Fatalf("cols=%d column %d row %d: packed %.8f, row layout %.8f",
						cols, c, r, state[r], want)
				}
			}
		}
	}
}

// The packed product against the row layout, on the shapes the E2B computes:
// the feed forward's two halves, one attention projection, and a batch of one.
func benchPacked(b *testing.B, rows, cols, size int, packed bool) {
	r := rand.New(rand.NewSource(5))
	w := make([]byte, rows*cols/QuantBlock*q4_0BlockBytes)
	for i := range w {
		w[i] = byte(r.Intn(256))
	}
	for i := 0; i < rows*cols/QuantBlock; i++ {
		w[i*q4_0BlockBytes+1] = byte(0x30 + r.Intn(2))
	}
	x := NewBatch(cols, size)
	for c := 0; c < size; c++ {
		for i := range x.F[c] {
			x.F[c][i] = r.Float32()*2 - 1
		}
	}
	x.Quantize()
	ys := make([][]float32, size)
	for c := range ys {
		ys[c] = make([]float32, rows)
	}
	m := Matrix{Data: w, Quant: Q4_0, Rows: rows, Cols: cols}
	if packed {
		m.Repack()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.MatVecBatch(x, ys)
	}
	b.ReportMetric(float64(rows)*float64(cols)*float64(size)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
}

func BenchmarkPackedFFN(b *testing.B)    { benchPacked(b, 12288, 1536, 64, true) }
func BenchmarkPackedDown(b *testing.B)   { benchPacked(b, 1536, 12288, 64, true) }
func BenchmarkPackedQ(b *testing.B)      { benchPacked(b, 2048, 1536, 64, true) }
func BenchmarkPackedSingle(b *testing.B) { benchPacked(b, 12288, 1536, 1, true) }
