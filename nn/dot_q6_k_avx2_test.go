//go:build amd64

package nn

import (
	"math/rand"
	"testing"
)

// The kernel and the portable form compute the same integers in the same
// order, so they agree bit for bit.
func TestDotQ6_KAVX2MatchesTheGoForm(t *testing.T) {
	if !avx2 {
		t.Skip("no AVX2")
	}
	r := rand.New(rand.NewSource(7))
	for _, superblocks := range []int{1, 2, 6} {
		n := superblocks * SuperBlock
		w := make([]byte, superblocks*q6_kBlockBytes)
		for i := range w {
			w[i] = byte(r.Intn(256))
		}
		x := NewBatch(n, 1)
		for i := range x.F[0] {
			x.F[0][i] = r.Float32()*2 - 1
		}
		x.QuantizeK()

		want := dotQ6_KGo(w, x.QK, x.BSums, x.KScales, n)
		got := dotQ6_KAVX2(&w[0], &x.QK[0], &x.BSums[0], &x.KScales[0], n)
		if got != want {
			t.Fatalf("%d superblocks: %v against %v", superblocks, got, want)
		}
	}
}
