package vk

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestSplitQ6_KHoldsTheSameWeights reads a packed row back the way the kernels
// read it and holds it to nn's reference dequantiser, exactly.
//
// Nothing is folded here either: the fp16 magnitude and the sixteen signed
// group scales are the file's own bytes moved, and a block is the file's own
// magnitudes reordered. Q6_K is the one of the three whose order is genuinely
// awkward — four consecutive weights come from two bytes four apart and two bit
// positions of a third — so this test is the whole of the argument that the
// reordering is the same walk nn does.
func TestSplitQ6_KHoldsTheSameWeights(t *testing.T) {
	const rows = 3
	for _, cols := range []int{1024, 3840} {
		t.Run(itoa(cols), func(t *testing.T) { splitQ6KHoldsTheSameWeights(t, rows, cols) })
	}
}

// 3840 columns is fifteen superblocks, an odd count, which is where the row
// padding rowBytesQ6_K applies and where an unpadded layout put every other row
// off a word.
func splitQ6KHoldsTheSameWeights(t *testing.T, rows, cols int) {
	const superBytes = 210
	nsb := cols / nn.SuperBlock

	rng := rand.New(rand.NewSource(6))
	src := make([]byte, rows*nsb*superBytes)
	rng.Read(src)
	for r := 0; r < rows; r++ {
		for sb := 0; sb < nsb; sb++ {
			at := (r*nsb+sb)*superBytes + 208
			binary.LittleEndian.PutUint16(src[at:], nn.Narrow(rng.Float32()*0.05))
		}
	}

	packed := splitQ6_K(src, rows, cols)
	if got, want := len(packed), rows*rowBytesQ6_K(cols); got != want {
		t.Fatalf("packed length %d, want %d", got, want)
	}

	nb := cols / nn.QuantBlock
	want := make([]float32, cols)
	for r := 0; r < rows; r++ {
		nn.DequantizeQ6_K(src[r*nsb*superBytes:(r+1)*nsb*superBytes], cols, want)
		stride := rowBytesQ6_K(cols)
		row := packed[r*stride : (r+1)*stride]
		blocks := row[q6kBlockBase(nsb):]
		for b := 0; b < nb; b++ {
			sb := b / 8
			d := nn.Widen(binary.LittleEndian.Uint16(row[16*nsb+2*sb:]))
			// Two group scales to a block of thirty-two, sixteen weights each.
			s := [2]int8{int8(row[16*sb+2*(b%8)]), int8(row[16*sb+2*(b%8)+1])}

			pack := blocks[b*24 : b*24+24]
			hi := [2]uint32{binary.LittleEndian.Uint32(pack[0:]), binary.LittleEndian.Uint32(pack[4:])}
			for j := 0; j < 16; j++ {
				for half := 0; half < 2; half++ {
					l := j + 16*half
					q := int32(pack[8+j]>>uint(4*half)&0x0F) | int32(hi[half]>>uint(2*j)&3)<<4
					got := d * float32(s[l/16]) * float32(q-32)
					if got != want[b*32+l] {
						t.Fatalf("row %d block %d weight %d: packed %g, reference %g",
							r, b, l, got, want[b*32+l])
					}
				}
			}
		}
	}
}
