package nn

import (
	"runtime"
	"sync"
	"testing"
)

func BenchmarkMatVecFF(b *testing.B) {
	const s, e = 4096, 1024
	w := make([]byte, s*e*2)
	for i := range w {
		w[i] = byte(i)
	}
	x := make([]float32, e)
	y := make([]float32, s)
	b.SetBytes(int64(s * e * 2))
	for i := 0; i < b.N; i++ {
		MatVecBF16(w, x, s, e, y)
	}
}

func BenchmarkMatVecFF1T(b *testing.B) {
	const s, e = 512, 1024
	w := make([]byte, s*e*2)
	x := make([]float32, e)
	y := make([]float32, s)
	b.SetBytes(int64(s * e * 2))
	for i := 0; i < b.N; i++ {
		MatVecBF16(w, x, s, e, y)
	}
}

var sink uint64

// BenchmarkParallelRead gives an order of magnitude for the memory ceiling:
// the same number of bytes, simply read and summed.
func BenchmarkParallelRead(b *testing.B) {
	const n = 4096 * 1024
	w := make([]uint16, n)
	b.SetBytes(int64(n * 2))
	for i := 0; i < b.N; i++ {
		nb := runtime.NumCPU()
		par := len(w) / nb
		var wg sync.WaitGroup
		var mu sync.Mutex
		for p := 0; p < nb; p++ {
			wg.Add(1)
			go func(d, f int) {
				defer wg.Done()
				var s uint64
				for _, v := range w[d:f] {
					s += uint64(v)
				}
				mu.Lock()
				sink += s
				mu.Unlock()
			}(p*par, min((p+1)*par, len(w)))
		}
		wg.Wait()
	}
}

// BenchmarkMatVecRAM: matrix too large for the cache, like the flow_lm that
// re-reads 604 MB of weights on every frame.
func BenchmarkMatVecRAM(b *testing.B) {
	const s, e = 51200, 1024
	w := make([]byte, s*e*2)
	x := make([]float32, e)
	y := make([]float32, s)
	b.SetBytes(int64(s * e * 2))
	for i := 0; i < b.N; i++ {
		MatVecBF16(w, x, s, e, y)
	}
}

func BenchmarkDotQ4_0(b *testing.B) {
	const n = 1536
	w := make([]byte, n/QuantBlock*q4_0BlockBytes)
	x := NewBatch(n, 1)
	x.Quantize()
	b.SetBytes(int64(len(w)))
	b.ResetTimer()
	var state [9]float32
	for i := 0; i < b.N; i++ {
		dotQ4_0(w, x, 0, 0, n, state[:], Begin|Finish)
	}
}
