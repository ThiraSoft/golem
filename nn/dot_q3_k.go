package nn

// The Q3_K product, in integers.
//
// This is dot_q6_k.go's arithmetic at three bits. The magnitudes stay where
// they are, unsigned in 0..7, and multiply a Q8_K activation as small integers;
// each group of sixteen weights carries a signed scale, and the recentring by
// four is not applied to the weights at all — it comes out as 4 x scale x sum(q)
// per group, which is why the activation carries its group sums.
//
// What it replaces is the same thing Q6_K's replaced: a whole row expanded into
// float32 and multiplied a lane at a time. A Q3_K_S is that format for two
// hundred and fifty-two of its tensors, so on the processor it was the whole
// model going through the slowest path in nn.
//
// The recentring cannot ride the block's scale the way Q4_0's eight does,
// because the two halves of a block of thirty-two carry different scales. That
// is the same reason vk/split_q3k.go gives for leaving it out of the card's
// packing, and it is why both readers pay it against the activation's own sums.

import "encoding/binary"

// dotQ3_KGo is the portable form. n must be a multiple of SuperBlock.
func dotQ3_KGo(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	var total float32
	for b := 0; b < n/SuperBlock; b++ {
		block := w[b*q3_kBlockBytes : (b+1)*q3_kBlockBytes]
		hmask, qs, sc := block[0:32], block[32:96], block[96:108]
		superScale := halfToFloat(binary.LittleEndian.Uint16(block[108:]))
		groupScales := q3_kScales(sc)
		activations := q[b*SuperBlock : (b+1)*SuperBlock]
		groupSums := bsums[b*(SuperBlock/16) : (b+1)*(SuperBlock/16)]

		var sum int32
		// The file's own traversal, which is not the matrix's: the four blocks
		// of a window of qs are four bit positions of the same thirty-two
		// bytes, and their third bits are four bit positions of the same
		// thirty-two bytes of hmask. Walking it this way reads each of those
		// bytes once for four weights.
		for j2 := 0; j2 < 8; j2++ {
			qSub := qs[(j2/4)*32:]
			shift := uint(2 * (j2 % 4))
			for half := 0; half < 2; half++ {
				scale := int32(groupScales[2*j2+half])
				a := activations[j2*32+half*16:]
				for l := 0; l < 16; l++ {
					q2 := int32(qSub[half*16+l]>>shift) & 3
					hbit := int32(hmask[half*16+l]>>uint(j2)) & 1
					sum += scale * (q2 | hbit<<2) * int32(a[l])
				}
			}
		}
		// The recentring, group by group: the weights were read shifted by four.
		var shift int32
		for g, groupSum := range groupSums {
			shift += int32(groupScales[g]) * int32(groupSum)
		}
		total += float32(sum-4*shift) * superScale * scales[b]
	}
	return total
}

// MatVecQ3_K computes y = W*x for Q3_K weights. x must already carry its Q8_K
// form.
func MatVecQ3_K(w []byte, b *Batch, outputs, inputs int, ys [][]float32) {
	if inputs%SuperBlock != 0 {
		panic("nn: Q3_K rows must be a multiple of the superblock size")
	}
	if b.QK == nil {
		panic("nn: a Q3_K product needs the activation in its Q8_K form")
	}
	InParallel(outputs, outputs*inputs*b.Size, func(start, end int) {
		matVecQ3_KRows(w, b, inputs, ys, start, end)
	})
}

// matVecQ3_KRows computes rows [start, end) on the caller's thread.
func matVecQ3_KRows(w []byte, b *Batch, inputs int, ys [][]float32, start, end int) {
	stride := inputs / SuperBlock * q3_kBlockBytes
	for r := start; r < end; r++ {
		row := w[r*stride : (r+1)*stride]
		for c := 0; c < b.Size; c++ {
			ys[c][r] = dotQ3_K(row,
				b.QK[c*b.Width:], b.BSums[c*b.Width/16:], b.KScales[c*b.Width/SuperBlock:], inputs)
		}
	}
}
