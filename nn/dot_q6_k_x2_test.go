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
	// Seeded, so a failure can be reproduced rather than waited for.
	rng := rand.New(rand.NewPCG(1, 2))
	w := make([]byte, n/SuperBlock*q6_kBlockBytes)
	for i := range w {
		w[i] = byte(rng.Uint32())
	}
	// A block's last two bytes are its fp16 scale, and random bytes land on a
	// NaN or an infinity about six times in a hundred. Both kernels then return
	// NaN, and NaN equals nothing — not even itself — so the comparison below
	// failed about one run in eight for a reason that had nothing to do with
	// the kernels. The scales are given a value a quantizer would have written.
	for blk := 0; blk*q6_kBlockBytes < len(w); blk++ {
		d := w[(blk+1)*q6_kBlockBytes-2:]
		h := floatToHalf(float32(blk+1) * 0.01)
		d[0], d[1] = byte(h), byte(h>>8)
	}
	b := NewBatch(n, 2)
	for c := 0; c < 2; c++ {
		for i := range b.F[c] {
			b.F[c][i] = rng.Float32()*4 - 2
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
