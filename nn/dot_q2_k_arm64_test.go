//go:build arm64

package nn

import (
	"math/rand"
	"testing"
)

// The NEON kernel against the portable form, exactly. The lengths cross the
// batch the kernel is called in, because the loop that fills the unpacked
// scales is per batch and an off-by-one there would leave the last superblock
// reading the previous one's.
func TestDotQ2_KNEONMatchesTheGoFormExactly(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(19))

	for _, superblocks := range []int{1, 2, 3, 8, sumsPerCall, sumsPerCall + 1, 2*sumsPerCall + 5} {
		n := superblocks * SuperBlock
		w := make([]byte, superblocks*q2_kBlockBytes)
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		for b := 0; b < superblocks; b++ {
			w[b*q2_kBlockBytes+81] = byte(0x2C + rng.Intn(3))
			w[b*q2_kBlockBytes+83] = byte(0x28 + rng.Intn(3))
		}
		x := NewBatch(n, 1)
		for i := range x.F[0] {
			x.F[0][i] = rng.Float32()*2 - 1
		}
		x.QuantizeK()

		want := dotQ2_KGo(w, x.QK, x.BSums, x.KScales, n)
		got := dotQ2_KNEON(w, x.QK, x.BSums, x.KScales, n)
		if got != want {
			t.Fatalf("%d superblocks: NEON %v, portable %v", superblocks, got, want)
		}
	}
}
