package nn

// Q2_K: 256 weights in 84 bytes.
//
// The layout is scales[16], qs[64], d, dmin. Each of the sixteen scale bytes
// packs two four-bit unsigned values — a scale in the low nibble and a minimum
// in the high one — and covers sixteen weights, so a superblock's group k is
// weights 16k to 16k+15 and carries scales[k]. A weight is two bits, and its
// value is d*(scale)*q less dmin*(minimum): there is no recentring, because
// this format has a minimum the way Q4_K does rather than an offset the way
// Q3_K and Q6_K do.
//
// The two-bit quants are walked exactly as Q3_K's are — the same four windows
// of thirty-two bytes at the same four bit positions — which is not a
// coincidence but the same packing without the third bit. nn/dequant_q3_k.go
// says it the other way round.
//
// Nothing generates Q2_K here; it is read only.

import "encoding/binary"

// DequantizeQ2_K expands one row of n weights. out must hold n floats.
func DequantizeQ2_K(w []byte, n int, out []float32) {
	if n%SuperBlock != 0 {
		panic("nn: Q2_K rows must be a multiple of the superblock size")
	}
	for b := 0; b < n/SuperBlock; b++ {
		block := w[b*q2_kBlockBytes : (b+1)*q2_kBlockBytes]
		scales, qs := block[0:16], block[16:80]
		d := halfToFloat(binary.LittleEndian.Uint16(block[80:]))
		dmin := halfToFloat(binary.LittleEndian.Uint16(block[82:]))
		dst := out[b*SuperBlock : (b+1)*SuperBlock]

		for w := 0; w < SuperBlock; w++ {
			blk := w / 32
			sub := (w % 32) / 16
			l := w % 16
			q2 := int32(qs[(blk/4)*32+sub*16+l]>>uint(2*(blk%4))) & 3
			sc := scales[w/16]
			dst[w] = d*float32(sc&0x0F)*float32(q2) - dmin*float32(sc>>4)
		}
	}
}

// matVecQ2_KRows computes rows [start, end) of the product on the caller's
// thread, against the batch's Q8_K form.
func matVecQ2_KRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := cols / SuperBlock * q2_kBlockBytes
	for r := start; r < end; r++ {
		row := w[r*stride : (r+1)*stride]
		for c := 0; c < b.Size; c++ {
			ys[c][r] = dotQ2_K(row,
				b.QK[c*b.Width:], b.BSums[c*b.Width/16:], b.KScales[c*b.Width/SuperBlock:], cols)
		}
	}
}
