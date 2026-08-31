package nn

// Q3_K: 256 weights in 110 bytes.
//
// The layout is hmask[32], qs[64], scales[12], d. A weight's three-bit quant
// comes from two places: two bits from qs, packed four weights to a byte in
// four passes that shift by 0, 2, 4 and 6, and a third bit from hmask, one bit
// per weight. The quant is q2 | (hbit<<2), recentred by 4 — not the inverted
// reading the comment on this once said, which does not match what ggml's own
// dequantizer computes.
//
// Sixteen six-bit scales cover sixteen weights each, packed into twelve bytes:
// the low nibble of scales[i] (i<8) or scales[i-8]'s high nibble (i>=8) holds
// four bits, and two more come from scales[8:12], two bits at a time, chosen by
// which quarter of the sixteen the scale is in. This packing is ggml's; the
// fixture is what confirms the traversal order it unpacks into.
//
// This exists because golem could not read the format it is measured against.
// Nothing generates Q3_K here; it is read only.

import "encoding/binary"

// DequantizeQ3_K expands one row of n weights. out must hold n floats.
func DequantizeQ3_K(w []byte, n int, out []float32) {
	if n%SuperBlock != 0 {
		panic("nn: Q3_K rows must be a multiple of the superblock size")
	}
	for b := 0; b < n/SuperBlock; b++ {
		block := w[b*q3_kBlockBytes : (b+1)*q3_kBlockBytes]
		hmask := block[0:32]
		qs := block[32:96]
		sc := block[96:108]
		d := halfToFloat(binary.LittleEndian.Uint16(block[108:]))
		dst := out[b*SuperBlock : (b+1)*SuperBlock]

		var scales [16]int8
		for i := 0; i < 4; i++ {
			scales[i] = int8((sc[i]&0x0F)|((sc[8+i]>>0)&3)<<4) - 32
			scales[4+i] = int8((sc[4+i]&0x0F)|((sc[8+i]>>2)&3)<<4) - 32
			scales[8+i] = int8((sc[i]>>4)|((sc[8+i]>>4)&3)<<4) - 32
			scales[12+i] = int8((sc[4+i]>>4)|((sc[8+i]>>6)&3)<<4) - 32
		}

		// w in 0..255. j2 is which of the eight sixteen-weight groups (four
		// per 128-weight half); half picks the low or high sixteen bytes of
		// qs within that group's 32-byte window; l is the weight's place in
		// that sixteen. hmask does not move with the half of the superblock —
		// only the shift and the tested bit do, which is why its index never
		// exceeds 32.
		for w := 0; w < SuperBlock; w++ {
			j2 := w / 32
			local := w % 32
			half := local / 16
			l := local % 16
			shift := uint(2 * (j2 % 4))
			qByte := qs[(j2/4)*32+half*16+l]
			hmByte := hmask[half*16+l]
			q2 := int32((qByte >> shift) & 3)
			hbit := int32((hmByte >> uint(j2)) & 1)
			q := q2 | hbit<<2
			dst[w] = d * float32(scales[2*j2+half]) * float32(q-4)
		}
	}
}

// matVecQ3_KRows dequantizes each row and dots it against the batch, same as
// Q5_K's kernel: Q3_K has no packed dot product here, only the format golem
// has to read in order to know what it is being measured against.
func matVecQ3_KRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := cols / SuperBlock * q3_kBlockBytes
	rowBuf := make([]float32, cols)
	for r := start; r < end; r++ {
		rowBytes := w[r*stride : (r+1)*stride]
		DequantizeQ3_K(rowBytes, cols, rowBuf)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(rowBuf, b.F[c])
		}
	}
}
