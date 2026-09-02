package vk

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestSplitQ4_KHoldsTheSameWeights reads a packed row back the way the kernels
// read it — a superblock header where splitQ4_K left it, sixteen contiguous
// bytes of nibbles a block — and holds it to nn's reference dequantiser.
//
// The demand here is exactness, where Q5_K's twin test allows a part in a
// thousand. Nothing is folded and nothing is rounded: the header is the file's
// own sixteen bytes moved, and the nibbles are the file's own weights
// reordered. A packing that costs no bits should cost no accuracy either, and
// the equality is what says the reordering did not lose a weight to its
// neighbouring sub-block — which is the mistake both K-quant readers here made
// for the life of an engine.
func TestSplitQ4_KHoldsTheSameWeights(t *testing.T) {
	const rows, cols = 3, 1024
	const superBytes = 144
	nsb := cols / nn.SuperBlock

	rng := rand.New(rand.NewSource(4))
	src := make([]byte, rows*nsb*superBytes)
	rng.Read(src)
	// Plausible magnitudes: random fp16 bits are as often infinities as not.
	for r := 0; r < rows; r++ {
		for sb := 0; sb < nsb; sb++ {
			at := (r*nsb + sb) * superBytes
			binary.LittleEndian.PutUint16(src[at:], nn.Narrow(rng.Float32()*0.05))
			binary.LittleEndian.PutUint16(src[at+2:], nn.Narrow(rng.Float32()*0.02))
		}
	}

	packed := splitQ4_K(src, rows, cols)
	if got, want := len(packed), rows*rowBytesQ4_K(cols); got != want {
		t.Fatalf("packed length %d, want %d", got, want)
	}

	nb := cols / nn.QuantBlock
	want := make([]float32, cols)
	for r := 0; r < rows; r++ {
		nn.DequantizeQ4_K(src[r*nsb*superBytes:(r+1)*nsb*superBytes], cols, want)
		row := packed[r*nb*18 : (r+1)*nb*18]
		nibbles := row[2*nb:]
		for b := 0; b < nb; b++ {
			// What a kernel does to find its scale: the superblock this block
			// belongs to, and its rank within it.
			head := row[16*(b/8) : 16*(b/8)+16]
			d := nn.Widen(binary.LittleEndian.Uint16(head[0:]))
			dmin := nn.Widen(binary.LittleEndian.Uint16(head[2:]))
			raw := head[4:16]
			i := b % 8
			var sc, m uint8
			if i < 4 {
				sc, m = raw[i]&63, raw[i+4]&63
			} else {
				sc = (raw[i+4] & 0xF) | ((raw[i-4] >> 6) << 4)
				m = (raw[i+4] >> 4) | ((raw[i] >> 6) << 4)
			}
			d1, m1 := d*float32(sc), dmin*float32(m)

			pack := nibbles[b*16 : b*16+16]
			for j := 0; j < 16; j++ {
				for half := 0; half < 2; half++ {
					l := j + 16*half
					q := (pack[j] >> uint(4*half)) & 0xF
					got := float32(q)*d1 - m1
					if got != want[b*32+l] {
						t.Fatalf("row %d block %d weight %d: packed %g, reference %g",
							r, b, l, got, want[b*32+l])
					}
				}
			}
		}
	}
}
