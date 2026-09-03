package vk

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestSplitQ2_KHoldsTheSameWeights reads a packed row back the way the kernels
// read it and holds it to nn's reference dequantiser, exactly.
//
// nn.DequantizeQ2_K is itself pinned to ggml's own output on a real tensor, so
// this is the join from the card's layout all the way to llama.cpp — the
// interleaving of qs, the order of the sixteen groups, and which nibble of a
// scale byte is the scale and which the minimum.
func TestSplitQ2_KHoldsTheSameWeights(t *testing.T) {
	const rows = 3
	for _, cols := range []int{1024, 3840} {
		t.Run(itoa(cols), func(t *testing.T) { splitQ2KHoldsTheSameWeights(t, rows, cols) })
	}
}

func splitQ2KHoldsTheSameWeights(t *testing.T, rows, cols int) {
	const superBytes = 84
	nsb := cols / nn.SuperBlock

	rng := rand.New(rand.NewSource(2))
	src := make([]byte, rows*nsb*superBytes)
	rng.Read(src)
	for r := 0; r < rows; r++ {
		for sb := 0; sb < nsb; sb++ {
			at := (r*nsb+sb)*superBytes + 80
			binary.LittleEndian.PutUint16(src[at:], nn.Narrow(rng.Float32()*0.05))
			binary.LittleEndian.PutUint16(src[at+2:], nn.Narrow(rng.Float32()*0.02))
		}
	}

	packed := splitQ2_K(src, rows, cols)
	if got, want := len(packed), rows*rowBytesQ2_K(cols); got != want {
		t.Fatalf("packed length %d, want %d", got, want)
	}

	nb := cols / nn.QuantBlock
	want := make([]float32, cols)
	for r := 0; r < rows; r++ {
		nn.DequantizeQ2_K(src[r*nsb*superBytes:(r+1)*nsb*superBytes], cols, want)
		stride := rowBytesQ2_K(cols)
		row := packed[r*stride : (r+1)*stride]
		blocks := row[q2kBlockBase(nsb):]
		for b := 0; b < nb; b++ {
			sb := b / 8
			d := nn.Widen(binary.LittleEndian.Uint16(row[16*nsb+4*sb:]))
			dmin := nn.Widen(binary.LittleEndian.Uint16(row[16*nsb+4*sb+2:]))
			// Two groups to a block of thirty-two, sixteen weights each, and a
			// group's byte holds its scale low and its minimum high.
			sc := [2]byte{row[16*sb+2*(b%8)], row[16*sb+2*(b%8)+1]}

			pack := blocks[b*8 : b*8+8]
			quants := [2]uint32{binary.LittleEndian.Uint32(pack[0:]), binary.LittleEndian.Uint32(pack[4:])}
			for j := 0; j < 16; j++ {
				for half := 0; half < 2; half++ {
					l := j + 16*half
					q := float32(quants[half] >> uint(2*j) & 3)
					got := d*float32(sc[half]&0x0F)*q - dmin*float32(sc[half]>>4)
					if got != want[b*32+l] {
						t.Fatalf("row %d block %d weight %d: packed %g, reference %g",
							r, b, l, got, want[b*32+l])
					}
				}
			}
		}
	}
}
