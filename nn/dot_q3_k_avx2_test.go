//go:build amd64

package nn

import (
	"math/rand"
	"testing"
)

// The kernel and the portable form compute the same integers in the same
// order, so they agree bit for bit.
//
// Random bytes are enough here and they would not be enough alone: what they
// cannot catch is a *scale* undone from the wrong bits, because a random
// twelve-byte header carries every six-bit value anyway and a wrong reading is
// still a plausible one. TestDotQ3_KMatchesTheDequantiser is the join to
// DequantizeQ3_K, which has a fixture behind it; this one holds the two
// implementations of the same walk together at several lengths.
func TestDotQ3_KAVX2MatchesTheGoForm(t *testing.T) {
	if !avx2 {
		t.Skip("no AVX2")
	}
	r := rand.New(rand.NewSource(7))
	for _, superblocks := range []int{1, 2, 6, 15} {
		n := superblocks * SuperBlock
		w := make([]byte, superblocks*q3_kBlockBytes)
		for i := range w {
			w[i] = byte(r.Intn(256))
		}
		x := NewBatch(n, 1)
		for i := range x.F[0] {
			x.F[0][i] = r.Float32()*2 - 1
		}
		x.QuantizeK()

		want := dotQ3_KGo(w, x.QK, x.BSums, x.KScales, n)
		got := dotQ3_KAVX2(&w[0], &x.QK[0], &x.BSums[0], &x.KScales[0], n)
		if got != want {
			t.Fatalf("%d superblocks: %v against %v", superblocks, got, want)
		}
	}
}
