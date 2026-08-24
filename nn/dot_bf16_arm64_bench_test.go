//go:build arm64

package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The bfloat16 kernel against the portable form it replaces.
//
// Read these as a ratio and never as a speed, for the reason set out at the top
// of dot_q4_0_arm64_bench_test.go: they are taken under emulation, where the
// cost of an instruction is the cost of translating it.
//
// The sizes are the two the speech engine actually asks for — the transformer's
// width and the widest projection in its flow net.
func benchBF16(b *testing.B, n int, neon bool) {
	rng := rand.New(rand.NewSource(5))
	row := make([]uint16, n)
	x := make([]float32, n)
	for i := range row {
		row[i] = uint16(math.Float32bits(rng.Float32()*4-2) >> 16)
		x[i] = rng.Float32()*2 - 1
	}
	var sink float32
	b.SetBytes(int64(n * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if neon {
			v, _ := fastDotBF16(&row[0], &x[0], n)
			sink = v
		} else {
			sink = scalarDotBF16(row, x)
		}
	}
	_ = sink
}

func BenchmarkDotBF16Go1024(b *testing.B)   { benchBF16(b, 1024, false) }
func BenchmarkDotBF16NEON1024(b *testing.B) { benchBF16(b, 1024, true) }
func BenchmarkDotBF16Go4096(b *testing.B)   { benchBF16(b, 4096, false) }
func BenchmarkDotBF16NEON4096(b *testing.B) { benchBF16(b, 4096, true) }
