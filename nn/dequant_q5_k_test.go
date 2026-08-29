package nn

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// ggmlDequantQ5K is ggml-quants.c's dequantize_row_q5_K, transcribed. It is
// the definition of the format; anything else is this engine's opinion of it.
func ggmlDequantQ5K(block []byte, out []float32) {
	d := halfToFloat(binary.LittleEndian.Uint16(block[0:2]))
	dmin := halfToFloat(binary.LittleEndian.Uint16(block[2:4]))
	scales := block[4:16]
	qh := block[16:48]
	ql := block[48:176]

	getScaleMin := func(j int) (sc, m uint8) {
		if j < 4 {
			return scales[j] & 63, scales[j+4] & 63
		}
		return (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4),
			(scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	}

	at, is, qlOff := 0, 0, 0
	u1, u2 := uint8(1), uint8(2)
	for j := 0; j < 256; j += 64 {
		sc, m := getScaleMin(is)
		d1, m1 := d*float32(sc), dmin*float32(m)
		sc, m = getScaleMin(is + 1)
		d2, m2 := d*float32(sc), dmin*float32(m)
		for l := 0; l < 32; l++ {
			h := float32(0)
			if qh[l]&u1 != 0 {
				h = 16
			}
			out[at] = d1*(float32(ql[qlOff+l]&0xF)+h) - m1
			at++
		}
		for l := 0; l < 32; l++ {
			h := float32(0)
			if qh[l]&u2 != 0 {
				h = 16
			}
			out[at] = d2*(float32(ql[qlOff+l]>>4)+h) - m2
			at++
		}
		qlOff += 32
		is += 2
		u1 <<= 2
		u2 <<= 2
	}
}

// This engine's Q5_K against the definition of Q5_K.
//
// It was written after the format had been misread for the whole life of the
// Qwen3.8 engine, on both paths at once: nn's dequantiser and vk's mat-vec
// made the same wrong assumption, so they agreed with each other and every
// test that compared one to the other passed. A hundred and twelve of every
// two hundred and fifty-six weights were somebody else's.
//
// The only thing that can catch that is the reference, transcribed rather than
// consulted, which is what ggmlDequantQ5K above is.
func TestQ5_KMatchesGGML(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	block := make([]byte, 176)
	for i := range block {
		block[i] = byte(r.Intn(256))
	}
	// A sane pair of halves rather than random bits.
	binary.LittleEndian.PutUint16(block[0:2], floatToHalf(0.01))
	binary.LittleEndian.PutUint16(block[2:4], floatToHalf(0.002))

	ours := make([]float32, 256)
	theirs := make([]float32, 256)
	DequantizeQ5_K(block, 256, ours)
	ggmlDequantQ5K(block, theirs)

	bad := 0
	worst := 0.0
	for i := range ours {
		d := math.Abs(float64(ours[i] - theirs[i]))
		if d > 1e-6 {
			bad++
			if d > worst {
				worst = d
			}
		}
	}
	if bad != 0 {
		t.Errorf("%d of 256 weights differ from ggml's own dequantiser, worst %g", bad, worst)
		for i := 0; i < 40; i++ {
			t.Logf("  w[%3d] ours=%9.5f ggml=%9.5f", i, ours[i], theirs[i])
		}
	}
}
