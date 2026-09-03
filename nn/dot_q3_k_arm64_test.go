//go:build arm64

package nn

import (
	"math/rand"
	"testing"
)

// The NEON kernel against the portable form, exactly.
//
// Every accumulation the assembly does is in integers, where it cannot lose
// anything and where the order does not matter, so the two forms reach the same
// number rather than a near one.
//
// The lengths cross the batch the kernel is called in — one superblock, a
// whole batch, a batch and one, two batches and a remainder — because the loop
// that fills the unpacked scales is per batch and an off-by-one there would
// leave the last superblock reading the previous one's.
func TestDotQ3_KNEONMatchesTheGoFormExactly(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(23))

	for _, superblocks := range []int{1, 2, 3, 8, sumsPerCall, sumsPerCall + 1, 2*sumsPerCall + 5} {
		n := superblocks * SuperBlock
		w := make([]byte, superblocks*q3_kBlockBytes)
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		// Keep the superblock scales to sane magnitudes: random fp16 bits can
		// be NaN, and a NaN compares unequal to itself.
		for b := 0; b < superblocks; b++ {
			w[b*q3_kBlockBytes+109] = byte(0x30 + rng.Intn(4))
		}

		x := NewBatch(n, 1)
		for i := range x.F[0] {
			x.F[0][i] = rng.Float32()*2 - 1
		}
		x.QuantizeK()

		want := dotQ3_KGo(w, x.QK, x.BSums, x.KScales, n)
		got := dotQ3_KNEON(w, x.QK, x.BSums, x.KScales, n)
		if got != want {
			t.Fatalf("%d superblocks: NEON %v, portable %v", superblocks, got, want)
		}
	}
}
