//go:build arm64

package nn

import "encoding/binary"

//go:noescape
func q3_kSumsNEON(w *byte, q *int8, scales *int8, blocks int, sums *int32)

// dotQ3_KNEON is dotQ3_KGo with the 256 products of each superblock done by
// SDOT. What is left in Go is what the assembly would gain nothing by doing:
// the six-bit scale unpacking, which is four masked shifts of three different
// bytes and is written once in q3_kScales; the recentring, which is sixteen
// multiplies against the activation's group sums; and the two float scales.
//
// The integer sum is exact in both forms, so the two agree bit for bit.
func dotQ3_KNEON(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	var sums [sumsPerCall]int32
	var unpacked [sumsPerCall * 16]int8
	var total float32

	blocks := n / SuperBlock
	for done := 0; done < blocks; done += sumsPerCall {
		batch := min(sumsPerCall, blocks-done)
		for i := 0; i < batch; i++ {
			block := w[(done+i)*q3_kBlockBytes : (done+i+1)*q3_kBlockBytes]
			s := q3_kScales(block[96:108])
			copy(unpacked[i*16:], s[:])
		}
		q3_kSumsNEON(&w[done*q3_kBlockBytes], &q[done*SuperBlock], &unpacked[0], batch, &sums[0])

		for i := 0; i < batch; i++ {
			b := done + i
			block := w[b*q3_kBlockBytes : (b+1)*q3_kBlockBytes]
			superScale := halfToFloat(binary.LittleEndian.Uint16(block[108:]))
			groupSums := bsums[b*(SuperBlock/16) : (b+1)*(SuperBlock/16)]

			// The recentring, group by group: the weights were read shifted
			// by four.
			var shift int32
			for g, groupSum := range groupSums {
				shift += int32(unpacked[i*16+g]) * int32(groupSum)
			}
			total += float32(sums[i]-4*shift) * superScale * scales[b]
		}
	}
	return total
}

func dotQ3_K(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	if dotprod && n%SuperBlock == 0 {
		return dotQ3_KNEON(w, q, bsums, scales, n)
	}
	return dotQ3_KGo(w, q, bsums, scales, n)
}
