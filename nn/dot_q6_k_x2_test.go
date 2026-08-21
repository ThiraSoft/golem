package nn

import (
	"math/rand/v2"
	"testing"
)

// The two-column product against the one-column one, bit for bit. The whole
// claim of the wider kernel is that it does the same arithmetic in the same
// order and only reads the row once; anything less exact would be a different
// engine, not a faster one.
func TestQ6KTwoColumnsAgreeWithOne(t *testing.T) {
	const n = 2 * SuperBlock
	w := make([]byte, n/SuperBlock*q6_kBlockBytes)
	for i := range w {
		w[i] = byte(rand.Uint32())
	}
	b := NewBatch(n, 2)
	for c := 0; c < 2; c++ {
		for i := range b.F[c] {
			b.F[c][i] = rand.Float32()*4 - 2
		}
		b.QuantizeColumnRange(c, 0, n)
	}
	b.QuantizeK()

	want := [2]float32{
		dotQ6_K(w, b.QK[0:], b.BSums[0:], b.KScales[0:], n),
		dotQ6_K(w, b.QK[b.Width:], b.BSums[b.Width/16:], b.KScales[b.Width/SuperBlock:], n),
	}
	var got [2]float32
	if !dotQ6_Kx2(w, b.QK[0:], b.QK[b.Width:], b.BSums[0:], b.BSums[b.Width/16:],
		b.KScales[0:], b.KScales[b.Width/SuperBlock:], n, &got) {
		t.Skip("no AVX2 on this machine")
	}
	if got != want {
		t.Errorf("two columns gave %v, one at a time gives %v", got, want)
	}
}
