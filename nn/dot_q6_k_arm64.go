//go:build arm64

package nn

import "encoding/binary"

//go:noescape
func q6_kSumsNEON(w *byte, q *int8, blocks int, sums *int32)

// dotQ6_KNEON is dotQ6_KGo with the 256 products of each superblock done by
// SDOT. What is left in Go is what the assembly would need two more hand-
// encoded instructions to do and would gain nothing by doing: the recentring,
// which is sixteen multiplies against the activation's group sums, and the two
// float scales.
//
// The integer sum is exact in both forms, so the two agree bit for bit.
func dotQ6_KNEON(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	var sums [sumsPerCall]int32
	var total float32

	blocks := n / SuperBlock
	for done := 0; done < blocks; done += sumsPerCall {
		batch := min(sumsPerCall, blocks-done)
		q6_kSumsNEON(&w[done*q6_kBlockBytes], &q[done*SuperBlock], batch, &sums[0])

		for i := 0; i < batch; i++ {
			b := done + i
			block := w[b*q6_kBlockBytes : (b+1)*q6_kBlockBytes]
			blockScales := block[192:208]
			superScale := halfToFloat(binary.LittleEndian.Uint16(block[208:]))
			groupSums := bsums[b*(SuperBlock/16) : (b+1)*(SuperBlock/16)]

			// The recentring, group by group: the weights were read shifted
			// by 32.
			var shift int32
			for g, groupSum := range groupSums {
				shift += int32(int8(blockScales[g])) * int32(groupSum)
			}
			total += float32(sums[i]-32*shift) * superScale * scales[b]
		}
	}
	return total
}

// dotQ6_Kx2 is the same product for two activations. The AVX2 kernel unpacks
// the row once for the two; this one does not, and reports false so the caller
// takes them one at a time. Whether unpacking once pays here is a question
// about a machine this was not written on.
func dotQ6_Kx2(w []byte, q0, q1 []int8, bsums0, bsums1 []int16, scales0, scales1 []float32, n int, out *[2]float32) bool {
	return false
}

func dotQ6_K(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	if dotprod && n%SuperBlock == 0 {
		return dotQ6_KNEON(w, q, bsums, scales, n)
	}
	return dotQ6_KGo(w, q, bsums, scales, n)
}
