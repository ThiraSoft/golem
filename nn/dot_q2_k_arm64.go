//go:build arm64

package nn

import "encoding/binary"

//go:noescape
func q2_kSumsNEON(w *byte, q *int8, scales *int8, blocks int, sums *int32)

// dotQ2_KNEON is dotQ2_KGo with the 256 products of each superblock done by
// SDOT. What stays in Go is the half that has no weights in it: the sixteen
// four-bit minima against the activation's sixteen group sums, and the two
// float scales.
//
// The integer sum is exact in both forms, so the two agree bit for bit.
func dotQ2_KNEON(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	var sums [sumsPerCall]int32
	var unpacked [sumsPerCall * 16]int8
	var total float32

	blocks := n / SuperBlock
	for done := 0; done < blocks; done += sumsPerCall {
		batch := min(sumsPerCall, blocks-done)
		for i := 0; i < batch; i++ {
			block := w[(done+i)*q2_kBlockBytes : (done+i+1)*q2_kBlockBytes]
			for g := 0; g < 16; g++ {
				unpacked[i*16+g] = int8(block[g] & 0x0F)
			}
		}
		q2_kSumsNEON(&w[done*q2_kBlockBytes], &q[done*SuperBlock], &unpacked[0], batch, &sums[0])

		for i := 0; i < batch; i++ {
			b := done + i
			block := w[b*q2_kBlockBytes : (b+1)*q2_kBlockBytes]
			d := halfToFloat(binary.LittleEndian.Uint16(block[80:]))
			dmin := halfToFloat(binary.LittleEndian.Uint16(block[82:]))
			groupSums := bsums[b*(SuperBlock/16) : (b+1)*(SuperBlock/16)]

			var mins int32
			for g, groupSum := range groupSums {
				mins += int32(block[g]>>4) * int32(groupSum)
			}
			total += (float32(sums[i])*d - float32(mins)*dmin) * scales[b]
		}
	}
	return total
}

func dotQ2_K(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	if dotprod && n%SuperBlock == 0 {
		return dotQ2_KNEON(w, q, bsums, scales, n)
	}
	return dotQ2_KGo(w, q, bsums, scales, n)
}
