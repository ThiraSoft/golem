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
// number rather than a near one. Q6_K is where a blind port is most likely to
// go wrong — six-bit weights split across two arrays, four planes at different
// offsets, a signed scale per group of sixteen — and an exact comparison is the
// sharpest instrument available for that.
func TestDotQ6_KNEONMatchesTheGoFormExactly(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(23))

	for _, superblocks := range []int{1, 2, 3, 8, sumsPerCall, sumsPerCall + 1, 2*sumsPerCall + 5} {
		n := superblocks * SuperBlock
		w := make([]byte, superblocks*q6_kBlockBytes)
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		// Keep the superblock scales to sane magnitudes: random fp16 bits can
		// be NaN, and a NaN compares unequal to itself.
		for b := 0; b < superblocks; b++ {
			w[b*q6_kBlockBytes+209] = byte(0x30 + rng.Intn(4))
		}

		x := NewBatch(n, 1)
		for i := range x.F[0] {
			x.F[0][i] = rng.Float32()*2 - 1
		}
		x.QuantizeK()

		want := dotQ6_KGo(w, x.QK, x.BSums, x.KScales, n)
		got := dotQ6_KNEON(w, x.QK, x.BSums, x.KScales, n)
		if got != want {
			t.Fatalf("%d superblocks: NEON %v, portable %v", superblocks, got, want)
		}
	}
}

// The extremes of the six-bit range, which random bytes reach only by accident:
// every weight at 0 and every weight at 63, against activations at both ends of
// the signed byte. A sign error in the high bits, or a mask that takes three
// bits where it should take two, shows here and nowhere else.
func TestDotQ6_KNEONAtTheEdgesOfTheRange(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	const superblocks = 2
	const n = superblocks * SuperBlock

	for _, w6 := range []byte{0, 1, 32, 62, 63} {
		for _, act := range []float32{-1, 1} {
			w := make([]byte, superblocks*q6_kBlockBytes)
			for b := 0; b < superblocks; b++ {
				block := w[b*q6_kBlockBytes : (b+1)*q6_kBlockBytes]
				// Low nibble in the low array, high two bits in the high array,
				// packed the way ggml packs them.
				for i := 0; i < 128; i++ {
					block[i] = w6&0x0F | (w6&0x0F)<<4
				}
				high := (w6 >> 4) & 3
				for i := 0; i < 64; i++ {
					block[128+i] = high | high<<2 | high<<4 | high<<6
				}
				for i := 0; i < 16; i++ {
					block[192+i] = byte(int8(i) - 8) // a spread of signed scales
				}
				block[208], block[209] = 0x00, 0x3C // 1.0 in fp16
			}

			x := NewBatch(n, 1)
			for i := range x.F[0] {
				x.F[0][i] = act
			}
			x.QuantizeK()

			want := dotQ6_KGo(w, x.QK, x.BSums, x.KScales, n)
			got := dotQ6_KNEON(w, x.QK, x.BSums, x.KScales, n)
			if got != want {
				t.Fatalf("weight %d, activation %v: NEON %v, portable %v", w6, act, got, want)
			}
		}
	}
}

// Through the dispatch, which is what the logit head actually calls.
func TestDotQ6_KNEONThroughTheDispatch(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(29))

	const superblocks = 6
	const n = superblocks * SuperBlock
	w := make([]byte, superblocks*q6_kBlockBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	for b := 0; b < superblocks; b++ {
		w[b*q6_kBlockBytes+209] = byte(0x30 + rng.Intn(4))
	}
	x := NewBatch(n, 1)
	for i := range x.F[0] {
		x.F[0][i] = rng.Float32()*2 - 1
	}
	x.QuantizeK()

	want := dotQ6_KGo(w, x.QK, x.BSums, x.KScales, n)
	if got := dotQ6_K(w, x.QK, x.BSums, x.KScales, n); got != want {
		t.Fatalf("dispatch gave %v, portable %v", got, want)
	}
}
