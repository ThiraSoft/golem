package nn

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// ggmlDequantQ4K is ggml-quants.c's dequantize_row_q4_K, transcribed. It is the
// definition of the format; anything else is this engine's opinion of it.
func ggmlDequantQ4K(block []byte, out []float32) {
	d := halfToFloat(binary.LittleEndian.Uint16(block[0:2]))
	dmin := halfToFloat(binary.LittleEndian.Uint16(block[2:4]))
	scales := block[4:16]
	q := block[16:144]

	getScaleMin := func(j int) (sc, m uint8) {
		if j < 4 {
			return scales[j] & 63, scales[j+4] & 63
		}
		return (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4),
			(scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	}

	at, is, qOff := 0, 0, 0
	for j := 0; j < 256; j += 64 {
		sc, m := getScaleMin(is)
		d1, m1 := d*float32(sc), dmin*float32(m)
		sc, m = getScaleMin(is + 1)
		d2, m2 := d*float32(sc), dmin*float32(m)
		for l := 0; l < 32; l++ {
			out[at] = d1*float32(q[qOff+l]&0xF) - m1
			at++
		}
		for l := 0; l < 32; l++ {
			out[at] = d2*float32(q[qOff+l]>>4) - m2
			at++
		}
		qOff += 32
		is += 2
	}
}

// This engine's Q4_K against the definition of Q4_K.
//
// Q5_K had exactly this bug, and dequant_q5_k.go still carries the note about
// it: a sub-block read as sixteen consecutive bytes, low nibbles for its first
// sixteen weights and high nibbles for its last sixteen, when the format pairs
// two sub-blocks over the same thirty-two bytes and gives the low nibbles to
// one and the high nibbles to the other. Q5_K was fixed because a test held it
// to the reference. Q4_K had no test, so it kept the bug — and nothing noticed,
// because golem's own models are Q4_0 and the K-quant path was never read.
func TestQ4_KMatchesGGML(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	block := make([]byte, 144)
	for i := range block {
		block[i] = byte(r.Intn(256))
	}
	binary.LittleEndian.PutUint16(block[0:2], floatToHalf(0.01))
	binary.LittleEndian.PutUint16(block[2:4], floatToHalf(0.002))

	ours := make([]float32, 256)
	theirs := make([]float32, 256)
	DequantizeQ4_K(block, 256, ours)
	ggmlDequantQ4K(block, theirs)

	bad, worst := 0, 0.0
	for i := range ours {
		if d := math.Abs(float64(ours[i] - theirs[i])); d > 1e-6 {
			bad++
			if d > worst {
				worst = d
			}
		}
	}
	if bad != 0 {
		t.Errorf("%d of 256 weights differ from ggml's own dequantiser, worst %g", bad, worst)
		for i := 0; i < 8; i++ {
			t.Logf("  w[%3d] ours=%9.5f ggml=%9.5f", i, ours[i], theirs[i])
		}
	}
}
