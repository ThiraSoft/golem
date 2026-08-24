//go:build arm64

package nn

import (
	"math/rand"
	"testing"
)

// The kernel against the portable form it replaces.
//
// Read these as a ratio and never as a speed. They are taken under emulation,
// where every instruction is translated before it runs, so the absolute figures
// belong to QEMU and not to any processor. What the ratio does say is roughly
// how much less work the kernel asks for — a translator's cost tracks the
// instruction count more closely than a real core's does, because it has none
// of the out-of-order machinery that makes a real core's cost depend on what
// surrounds an instruction rather than on how many there are.
//
// So: a ratio near one means the kernel is not doing less work and something is
// wrong. A large ratio does not promise the same on hardware, where the memory
// bus will take over long before the arithmetic does.
func benchQ4_0(b *testing.B, n int, neon bool) {
	if neon && !dotprod {
		b.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(3))
	blocks := n / QuantBlock
	w := make([]byte, blocks*q4_0BlockBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	for k := 0; k < blocks; k++ {
		w[k*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
	}
	batch := NewBatch(n, 1)
	for i := range batch.F[0] {
		batch.F[0][i] = rng.Float32()*2 - 1
	}
	batch.Quantize()

	var state [9]float32
	b.SetBytes(int64(len(w)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		state[0], state[8] = 0, 0
		if neon {
			dotQ4_0NEON(w, batch, 0, 0, n, state[:])
		} else {
			dotQ4_0Go(w, batch, 0, 0, n, state[:])
		}
	}
}

func BenchmarkDotQ4_0Go1024(b *testing.B)   { benchQ4_0(b, 1024, false) }
func BenchmarkDotQ4_0NEON1024(b *testing.B) { benchQ4_0(b, 1024, true) }
func BenchmarkDotQ4_0Go4096(b *testing.B)   { benchQ4_0(b, 4096, false) }
func BenchmarkDotQ4_0NEON4096(b *testing.B) { benchQ4_0(b, 4096, true) }
