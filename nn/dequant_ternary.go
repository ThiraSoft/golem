package nn

// PQ2_0 and PTQ1_0: Prism ternary weight formats at group size 128.
//
// PQ2_0 packs 128 two-bit weights into 34 bytes: one fp16 scale followed by
// 32 bytes of 2-bit quants.
//
// PTQ1_0 packs 128 base-3 trits into 28 bytes: 24 bytes qs (120 trits),
// 2 bytes qh (8 trits), and one fp16 scale at the end.

import "encoding/binary"

var (
	ptq1_0_stages = [3]int{32, 16, 8}
	pow3          = [6]uint8{1, 3, 9, 27, 81, 243}
)

// DequantizePQ2_0 expands one row of n weights. out must hold n floats.
func DequantizePQ2_0(w []byte, n int, out []float32) {
	if n%TernaryBlock != 0 {
		panic("nn: PQ2_0 rows must be a multiple of the ternary block size")
	}
	nb := n / TernaryBlock
	for i := 0; i < nb; i++ {
		block := w[i*pq2_0BlockBytes : (i+1)*pq2_0BlockBytes]
		d := halfToFloat(binary.LittleEndian.Uint16(block[:2]))
		qs := block[2:34]
		dst := out[i*TernaryBlock : (i+1)*TernaryBlock]

		for j := 0; j < TernaryBlock; j++ {
			byteIndex := j / 4
			bitOffset := (j % 4) * 2
			q := int((qs[byteIndex] >> bitOffset) & 0x03)
			// 00=-1, 01=0, 10=+1, 11=+2
			dst[j] = float32(q-1) * d
		}
	}
}

// DequantizePTQ1_0 expands one row of n weights. out must hold n floats.
func DequantizePTQ1_0(w []byte, n int, out []float32) {
	if n%TernaryBlock != 0 {
		panic("nn: PTQ1_0 rows must be a multiple of the ternary block size")
	}
	nb := n / TernaryBlock
	yIdx := 0
	for i := 0; i < nb; i++ {
		block := w[i*ptq1_0BlockBytes : (i+1)*ptq1_0BlockBytes]
		qs := block[:24]
		qh := block[24:26]
		d := halfToFloat(binary.LittleEndian.Uint16(block[26:28]))

		j := 0
		for s := 0; s < 3; s++ {
			c := ptq1_0_stages[s]
			for ; j+c <= len(qs); j += c {
				for n := 0; n < 5; n++ {
					for m := 0; m < c; m++ {
						q := qs[j+m] * pow3[n]
						xi := int16((uint16(q) * 3) >> 8)
						out[yIdx] = float32(xi-1) * d
						yIdx++
					}
				}
			}
		}
		for n := 0; n < 4; n++ {
			for h := 0; h < len(qh); h++ {
				q := qh[h] * pow3[n]
				xi := int16((uint16(q) * 3) >> 8)
				out[yIdx] = float32(xi-1) * d
				yIdx++
			}
		}
	}
}

// matVecPQ2_0Rows computes rows [start, end) of the product on the caller's thread,
// against the batch's float activations.
func matVecPQ2_0Rows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := cols / TernaryBlock * pq2_0BlockBytes
	var buf [TernaryBlock]float32
	for r := start; r < end; r++ {
		row := w[r*stride : (r+1)*stride]
		for c := 0; c < b.Size; c++ {
			ys[c][r] = 0
		}
		for blk := 0; blk < cols/TernaryBlock; blk++ {
			DequantizePQ2_0(row[blk*pq2_0BlockBytes:(blk+1)*pq2_0BlockBytes], TernaryBlock, buf[:])
			for c := 0; c < b.Size; c++ {
				ys[c][r] += DotF32(buf[:], b.F[c][blk*TernaryBlock:(blk+1)*TernaryBlock])
			}
		}
	}
}

// matVecPTQ1_0Rows computes rows [start, end) of the product on the caller's thread,
// against the batch's float activations.
func matVecPTQ1_0Rows(w []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	stride := cols / TernaryBlock * ptq1_0BlockBytes
	var buf [TernaryBlock]float32
	for r := start; r < end; r++ {
		row := w[r*stride : (r+1)*stride]
		for c := 0; c < b.Size; c++ {
			ys[c][r] = 0
		}
		for blk := 0; blk < cols/TernaryBlock; blk++ {
			DequantizePTQ1_0(row[blk*ptq1_0BlockBytes:(blk+1)*ptq1_0BlockBytes], TernaryBlock, buf[:])
			for c := 0; c < b.Size; c++ {
				ys[c][r] += DotF32(buf[:], b.F[c][blk*TernaryBlock:(blk+1)*TernaryBlock])
			}
		}
	}
}
