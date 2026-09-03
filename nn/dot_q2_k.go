package nn

// The Q2_K product, in integers.
//
// Two bits a weight, a four-bit scale and a four-bit minimum for every group of
// sixteen, and two fp16 that say what those nibbles are measured in. A weight
// is d*scale*q less dmin*minimum, so this is the one K-quant here whose
// correction is a *minimum* rather than an offset — Q3_K and Q6_K recentre
// their magnitudes, Q2_K subtracts a number that does not depend on the weight
// at all.
//
// That makes the second term cheaper here than anywhere else in this package.
// The minimum is constant over a group of sixteen, and a Q8_K activation
// already carries a sum over each group of sixteen — so the whole correction is
// one dot product of sixteen terms against numbers the caller brought, and no
// kernel has to touch the weights for it. On the card the same term costs a
// packed dot against a vector of ones, because there the activation's stored
// sum covers a block of thirty-two and the two halves have different minima.
//
// The magnitudes are walked exactly as Q3_K's are — the same four windows of
// thirty-two bytes at the same four bit positions — which is the same packing
// without the third bit.

import "encoding/binary"

// dotQ2_KGo is the portable form. n must be a multiple of SuperBlock.
func dotQ2_KGo(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	var total float32
	for b := 0; b < n/SuperBlock; b++ {
		block := w[b*q2_kBlockBytes : (b+1)*q2_kBlockBytes]
		groupScales, qs := block[0:16], block[16:80]
		d := halfToFloat(binary.LittleEndian.Uint16(block[80:]))
		dmin := halfToFloat(binary.LittleEndian.Uint16(block[82:]))
		activations := q[b*SuperBlock : (b+1)*SuperBlock]
		groupSums := bsums[b*(SuperBlock/16) : (b+1)*(SuperBlock/16)]

		var sum, mins int32
		for g := 0; g < 16; g++ {
			sc := groupScales[g]
			// The window and the shift are the block of thirty-two's, and the
			// half of it is which sixteen bytes of that window: two groups
			// share a shift and are sixteen bytes apart.
			blk, sub := g/2, g%2
			qSub := qs[(blk/4)*32+sub*16:]
			shift := uint(2 * (blk % 4))
			a := activations[g*16:]

			var dot int32
			for l := 0; l < 16; l++ {
				dot += int32(qSub[l]>>shift&3) * int32(a[l])
			}
			sum += int32(sc&0x0F) * dot
			mins += int32(sc>>4) * int32(groupSums[g])
		}
		total += (float32(sum)*d - float32(mins)*dmin) * scales[b]
	}
	return total
}

// MatVecQ2_K computes y = W*x for Q2_K weights. x must already carry its Q8_K
// form.
func MatVecQ2_K(w []byte, b *Batch, outputs, inputs int, ys [][]float32) {
	if inputs%SuperBlock != 0 {
		panic("nn: Q2_K rows must be a multiple of the superblock size")
	}
	if b.QK == nil {
		panic("nn: a Q2_K product needs the activation in its Q8_K form")
	}
	InParallel(outputs, outputs*inputs*b.Size, func(start, end int) {
		matVecQ2_KRows(w, b, inputs, ys, start, end)
	})
}
