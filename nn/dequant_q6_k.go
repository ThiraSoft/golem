package nn

// Q6_K: 256 weights in 210 bytes.
//
// The layout is not a simple packing. A superblock holds 128 bytes of low
// nibbles, 64 bytes carrying two high bits for each weight, 16 signed scales
// covering 16 weights each, and one fp16 scale for the whole superblock. Six
// bits of magnitude are assembled from the two sources, recentred by 32, then
// multiplied by both scales.
//
// The superblock is traversed in halves of 128 weights, and within a half in
// four groups of 32 that draw their high bits from different positions of the
// same byte. This ordering is ggml's; the fixture is what confirms it.

import "encoding/binary"

// DequantizeQ6_K expands one row of n weights. out must hold n floats.
func DequantizeQ6_K(w []byte, n int, out []float32) {
	if n%SuperBlock != 0 {
		panic("nn: Q6_K rows must be a multiple of the superblock size")
	}
	for b := 0; b < n/SuperBlock; b++ {
		block := w[b*q6_kBlockBytes : (b+1)*q6_kBlockBytes]
		low := block[0:128]
		high := block[128:192]
		scales := block[192:208]
		superScale := halfToFloat(binary.LittleEndian.Uint16(block[208:]))
		dst := out[b*SuperBlock : (b+1)*SuperBlock]

		for half := 0; half < 2; half++ {
			l := low[half*64 : half*64+64]
			h := high[half*32 : half*32+32]
			s := scales[half*8 : half*8+8]
			d := dst[half*128 : half*128+128]

			for i := 0; i < 32; i++ {
				group := i / 16 // two scales per group of 32

				q1 := int32(l[i]&0x0F) | int32((h[i]>>0)&3)<<4
				q2 := int32(l[i+32]&0x0F) | int32((h[i]>>2)&3)<<4
				q3 := int32(l[i]>>4) | int32((h[i]>>4)&3)<<4
				q4 := int32(l[i+32]>>4) | int32((h[i]>>6)&3)<<4

				d[i+0] = superScale * float32(int8(s[group+0])) * float32(q1-32)
				d[i+32] = superScale * float32(int8(s[group+2])) * float32(q2-32)
				d[i+64] = superScale * float32(int8(s[group+4])) * float32(q3-32)
				d[i+96] = superScale * float32(int8(s[group+6])) * float32(q4-32)
			}
		}
	}
}
