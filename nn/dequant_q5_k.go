package nn

// Q5_K: 256 weights in 176 bytes.
//
// A superblock holds 2 fp16 scales (d, dmin), 12 bytes of 6-bit scales and mins,
// 32 bytes carrying the 5th bit for each weight (qh), and 128 bytes containing
// the low 4 bits of the 256 weights (qs).

import "encoding/binary"

// DequantizeQ5_K expands one row of n weights. out must hold n floats.
func DequantizeQ5_K(w []byte, n int, out []float32) {
	if n%SuperBlock != 0 {
		panic("nn: Q5_K rows must be a multiple of the superblock size (256)")
	}
	blocks := n / SuperBlock
	for b := 0; b < blocks; b++ {
		block := w[b*q5_kBlockBytes : (b+1)*q5_kBlockBytes]
		d := halfToFloat(binary.LittleEndian.Uint16(block[0:2]))
		dmin := halfToFloat(binary.LittleEndian.Uint16(block[2:4]))
		scalesRaw := block[4:16]
		qh := block[16:48]
		qs := block[48:176]
		dst := out[b*SuperBlock : (b+1)*SuperBlock]

		var sc [8]uint8
		var m [8]uint8
		for j := 0; j < 4; j++ {
			sc[j] = scalesRaw[j] & 63
			m[j] = scalesRaw[j+4] & 63
		}
		for j := 4; j < 8; j++ {
			sc[j] = (scalesRaw[j+4] & 0xF) | ((scalesRaw[j-4] >> 6) << 4)
			m[j] = (scalesRaw[j+4] >> 4) | ((scalesRaw[j] >> 6) << 4)
		}

		// Two sub-blocks share thirty-two bytes of qs: the even one takes
		// their low nibbles and the odd one their high nibbles, both over the
		// same thirty-two positions. The fifth bit is bit i of qh's thirty-two
		// bytes, one byte a position.
		//
		// This used to read sixteen consecutive bytes a sub-block and take the
		// low nibbles for its first sixteen weights and the high nibbles for
		// its last sixteen, which is a different sixteen weights and a
		// different bit of qh. Half of every sub-block was somebody else's.
		// ggml-quants.c's dequantize_row_q5_K is the definition and
		// TestQ5_KMatchesGGML holds this to it.
		for i := 0; i < 8; i++ {
			d1 := d * float32(sc[i])
			m1 := dmin * float32(m[i])
			qSub := qs[(i/2)*32 : (i/2)*32+32]
			shift := uint(4 * (i & 1))
			qhBit := uint8(1 << i)

			for l := 0; l < 32; l++ {
				h := uint8(0)
				if qh[l]&qhBit != 0 {
					h = 16
				}
				q := ((qSub[l] >> shift) & 0xF) | h
				dst[i*32+l] = float32(q)*d1 - m1
			}
		}
	}
}

// matVecQ5_KRows computes y = W * x for Q5_K weights against float32 activations.
func matVecQ5_KRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := cols / SuperBlock * q5_kBlockBytes
	rowBuf := make([]float32, cols)
	for r := start; r < end; r++ {
		rowBytes := w[r*stride : (r+1)*stride]
		DequantizeQ5_K(rowBytes, cols, rowBuf)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(rowBuf, b.F[c])
		}
	}
}
