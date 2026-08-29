package nn

// Q4_K: 256 weights in 144 bytes.
//
// A superblock holds 2 fp16 scales (d, dmin), 12 bytes of 6-bit scales and mins,
// and 128 bytes containing 256 4-bit weights.
//
// Layout:
// - d: 2 bytes (fp16)
// - dmin: 2 bytes (fp16)
// - scales: 12 bytes (6 bytes for 8 scales, 6 bytes for 8 mins)
// - qs: 128 bytes (8 chunks of 16 bytes = 32 weights per chunk)

import "encoding/binary"

// DequantizeQ4_K expands one row of n weights. out must hold n floats.
func DequantizeQ4_K(w []byte, n int, out []float32) {
	if n%SuperBlock != 0 {
		panic("nn: Q4_K rows must be a multiple of the superblock size (256)")
	}
	blocks := n / SuperBlock
	for b := 0; b < blocks; b++ {
		block := w[b*q4_kBlockBytes : (b+1)*q4_kBlockBytes]
		d := halfToFloat(binary.LittleEndian.Uint16(block[0:2]))
		dmin := halfToFloat(binary.LittleEndian.Uint16(block[2:4]))
		scalesRaw := block[4:16]
		qs := block[16:144]
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
		// their low nibbles and the odd one their high nibbles, over the same
		// thirty-two positions and with its own scale and minimum.
		//
		// This used to read sixteen consecutive bytes a sub-block and give
		// both nibbles of each byte to the same scale, which is a different
		// sixteen weights and the wrong scale for half of them. Q5_K carried
		// the identical mistake until a test held it to the reference;
		// dequant_q5_k.go still says so. Q4_K had no test, so it kept it.
		// ggml-quants.c's dequantize_row_q4_K is the definition and
		// TestQ4_KMatchesGGML holds this to it.
		for i := 0; i < 8; i++ {
			d1 := d * float32(sc[i])
			m1 := dmin * float32(m[i])
			qSub := qs[(i/2)*32 : (i/2)*32+32]
			shift := uint(4 * (i & 1))
			base := i * 32
			for l := 0; l < 32; l++ {
				dst[base+l] = float32((qSub[l]>>shift)&0xF)*d1 - m1
			}
		}
	}
}

// matVecQ4_KRows computes y = W * x for Q4_K weights against float32 activations.
func matVecQ4_KRows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := cols / SuperBlock * q4_kBlockBytes
	rowBuf := make([]float32, cols)
	for r := start; r < end; r++ {
		rowBytes := w[r*stride : (r+1)*stride]
		DequantizeQ4_K(rowBytes, cols, rowBuf)
		for c := 0; c < b.Size; c++ {
			ys[c][r] = DotF32(rowBuf, b.F[c])
		}
	}
}
