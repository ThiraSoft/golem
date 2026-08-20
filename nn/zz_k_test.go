package nn

import (
	"math/rand"
	"testing"
)

func benchBatch(b *testing.B, rows, cols, size int) {
	r := rand.New(rand.NewSource(5))
	w := make([]byte, rows*cols/QuantBlock*q4_0BlockBytes)
	for i := range w {
		w[i] = byte(r.Intn(256))
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
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.MatVecBatch(x, ys)
	}
	b.ReportMetric(float64(rows)*float64(cols)*float64(size)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
}

func BenchmarkBatchFFN(b *testing.B)   { benchBatch(b, 12288, 1536, 64) }
func BenchmarkBatchDown(b *testing.B)  { benchBatch(b, 1536, 12288, 64) }
func BenchmarkBatchQ(b *testing.B)     { benchBatch(b, 2048, 1536, 64) }
func BenchmarkBatchSmall(b *testing.B) { benchBatch(b, 256, 1536, 64) }
