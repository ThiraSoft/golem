package vk

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestSplitQ5_KHoldsTheSameWeights reads a packed row back the way
// shaders/matmul_coop.comp does under -DQ5K and holds it to nn's reference
// dequantiser.
//
// This is the test the format did not have. Q5_K was read wrongly on both
// paths for the life of the engine — a hundred and twelve weights of every
// two hundred and fifty-six belonged to the neighbouring sub-block — and it
// survived because the processor and the card shared the error, so every
// comparison between them agreed. A packing is held to the reference or it is
// held to nothing.
func TestSplitQ5_KHoldsTheSameWeights(t *testing.T) {
	const rows, cols = 3, 1024
	const superBytes = 176
	nsb := cols / 256

	rng := rand.New(rand.NewSource(5))
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

	packed := splitQ5_K(src, rows, cols)
	if got, want := len(packed), rows*rowBytesQ5_K(cols); got != want {
		t.Fatalf("packed length %d, want %d", got, want)
	}

	nb := cols / nn.QuantBlock
	want := make([]float32, cols)
	for r := 0; r < rows; r++ {
		nn.DequantizeQ5_K(src[r*nsb*superBytes:(r+1)*nsb*superBytes], cols, want)
		row := packed[r*nb*24 : (r+1)*nb*24]
		highs := row[4*nb:]
		nibbles := row[8*nb:]
		for b := 0; b < nb; b++ {
			d1 := nn.Widen(binary.LittleEndian.Uint16(row[4*b:]))
			m1 := nn.Widen(binary.LittleEndian.Uint16(row[4*b+2:]))
			hi := binary.LittleEndian.Uint32(highs[4*b:])
			pack := nibbles[b*16 : b*16+16]
			for j := 0; j < 16; j++ {
				for half := 0; half < 2; half++ {
					l := j + 16*half
					q := (pack[j] >> uint(4*half)) & 0xF
					if hi&(1<<uint(l)) != 0 {
						q |= 16
					}
					got := float32(q)*d1 - m1
					// The reference multiplies out in fp32 from the file's own
					// magnitudes; this packing rounds d*sc and dmin*m to fp16
					// once, on the way to the card. So the demand is a part in
					// a thousand of the two terms rather than of their
					// difference — a weight near zero is the difference of two
					// numbers that are not, and it is the terms that carry the
					// rounding. Both operands the cooperative kernel stages
					// are fp16 anyway.
					scale := math.Abs(float64(float32(q)*d1)) + math.Abs(float64(m1))
					if e := math.Abs(float64(got - want[b*32+l])); e > 1e-3*scale+1e-6 {
						t.Fatalf("row %d block %d weight %d: packed %g, reference %g",
							r, b, l, got, want[b*32+l])
					}
				}
			}
		}
	}
}
