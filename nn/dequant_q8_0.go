package nn

import (
	"unsafe"
)

// matVecQ8_0Rows computes ys = W * b for rows in [start, end) of a Q8_0 matrix.
func matVecQ8_0Rows(data []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	nb := cols / QuantBlock         // 32
	rowBytes := nb * q8_0BlockBytes // 34

	for r := start; r < end; r++ {
		row := data[r*rowBytes : (r+1)*rowBytes]

		for c := 0; c < b.Size; c++ {
			var acc float32
			xF := b.F[c]

			for block := 0; block < nb; block++ {
				blk := row[block*q8_0BlockBytes : (block+1)*q8_0BlockBytes]
				dBits := *(*uint16)(unsafe.Pointer(&blk[0]))
				d := halfToFloat(dBits)
				qs := unsafe.Slice((*int8)(unsafe.Pointer(&blk[2])), 32)
				xSlice := xF[block*32 : (block+1)*32]

				var sum float32
				for i := 0; i < 32; i++ {
					sum += float32(qs[i]) * xSlice[i]
				}
				acc += sum * d
			}
			ys[c][r] = acc
		}
	}
}

// dequantizeQ8_0Row expands one Q8_0 row into out.
func dequantizeQ8_0Row(w []byte, n int, out []float32) {
	nb := n / QuantBlock
	for b := 0; b < nb; b++ {
		block := w[b*q8_0BlockBytes : (b+1)*q8_0BlockBytes]
		dBits := *(*uint16)(unsafe.Pointer(&block[0]))
		d := halfToFloat(dBits)
		qs := unsafe.Slice((*int8)(unsafe.Pointer(&block[2])), 32)
		dst := out[b*QuantBlock : (b+1)*QuantBlock]
		for i := 0; i < 32; i++ {
			dst[i] = float32(qs[i]) * d
		}
	}
}
