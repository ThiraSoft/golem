package vk

import (
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestSplitQ3_KHoldsTheSameWeights reads a packed row back the way the kernels
// read it and holds it to nn's reference dequantiser, exactly.
//
// It is the whole of the argument that splitQ3_K's reordering is the same walk
// nn does, and Q3_K is the format where that argument is most needed: four
// blocks share a window of qs at four bit positions, all eight share the same
// thirty-two bytes of third bits, and the sixteen scales come out of twelve
// bytes in an order that is ggml's alone. A reordering that got any one of the
// three wrong would still produce plausible numbers.
func TestSplitQ3_KHoldsTheSameWeights(t *testing.T) {
	const rows = 3
	for _, cols := range []int{1024, 3840} {
		t.Run(itoa(cols), func(t *testing.T) { splitQ3KHoldsTheSameWeights(t, rows, cols) })
	}
}

// 3840 columns is fifteen superblocks, an odd count, which is where the row
// padding rowBytesQ3_K applies: a hundred and fourteen bytes a superblock is
// not a word, so an unpadded layout would put every other row off one.
func splitQ3KHoldsTheSameWeights(t *testing.T, rows, cols int) {
	const superBytes = 110
	nsb := cols / nn.SuperBlock

	rng := rand.New(rand.NewSource(3))
	src := make([]byte, rows*nsb*superBytes)
	rng.Read(src)
	for r := 0; r < rows; r++ {
		for sb := 0; sb < nsb; sb++ {
			at := (r*nsb+sb)*superBytes + 108
			binary.LittleEndian.PutUint16(src[at:], nn.Narrow(rng.Float32()*0.05))
		}
	}

	packed := splitQ3_K(src, rows, cols)
	if got, want := len(packed), rows*rowBytesQ3_K(cols); got != want {
		t.Fatalf("packed length %d, want %d", got, want)
	}

	nb := cols / nn.QuantBlock
	want := make([]float32, cols)
	for r := 0; r < rows; r++ {
		nn.DequantizeQ3_K(src[r*nsb*superBytes:(r+1)*nsb*superBytes], cols, want)
		stride := rowBytesQ3_K(cols)
		row := packed[r*stride : (r+1)*stride]
		blocks := row[q3kBlockBase(nsb):]
		for b := 0; b < nb; b++ {
			sb := b / 8
			d := nn.Widen(binary.LittleEndian.Uint16(row[16*nsb+2*sb:]))
			// Two group scales to a block of thirty-two, sixteen weights each,
			// already less thirty-two where the file packs them six bits wide.
			s := [2]int8{int8(row[16*sb+2*(b%8)]), int8(row[16*sb+2*(b%8)+1])}

			pack := blocks[b*12 : b*12+12]
			low := [2]uint32{binary.LittleEndian.Uint32(pack[0:]), binary.LittleEndian.Uint32(pack[4:])}
			high := binary.LittleEndian.Uint32(pack[8:])
			for j := 0; j < 16; j++ {
				for half := 0; half < 2; half++ {
					l := j + 16*half
					q := int32(low[half]>>uint(2*j)&3) | int32(high>>uint(16*half+j)&1)<<2
					got := d * float32(s[half]) * float32(q-4)
					if got != want[b*32+l] {
						t.Fatalf("row %d block %d weight %d: packed %g, reference %g",
							r, b, l, got, want[b*32+l])
					}
				}
			}
		}
	}
}
