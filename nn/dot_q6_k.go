package nn

// The Q6_K product, in integers.
//
// The old path dequantized a whole row into floats and multiplied those: three
// quarters of a gigabyte of weights turned into three gigabytes of float32 on
// every token, for a product the machine then did one lane at a time. It was
// the single most expensive thing in the engine.
//
// This is what ggml does instead. The six-bit magnitudes stay where they are,
// unsigned in 0..63, and multiply a Q8_K activation as small integers. Each
// group of sixteen weights carries a signed scale, and the recentring by 32 is
// not applied to the weights at all: it comes out as 32 x scale x sum(q) per
// group, which is why the activation carries its group sums.

import "encoding/binary"

// dotQ6_KGo is the portable form. n must be a multiple of SuperBlock.
func dotQ6_KGo(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	var total float32
	for b := 0; b < n/SuperBlock; b++ {
		block := w[b*q6_kBlockBytes : (b+1)*q6_kBlockBytes]
		low := block[0:128]
		high := block[128:192]
		blockScales := block[192:208]
		superScale := halfToFloat(binary.LittleEndian.Uint16(block[208:]))
		activations := q[b*SuperBlock : (b+1)*SuperBlock]
		groupSums := bsums[b*(SuperBlock/16) : (b+1)*(SuperBlock/16)]

		var sum int32
		for half := 0; half < 2; half++ {
			l := low[half*64 : half*64+64]
			h := high[half*32 : half*32+32]
			s := blockScales[half*8 : half*8+8]
			a := activations[half*128 : half*128+128]

			for i := 0; i < 32; i++ {
				group := i / 16

				sum += int32(int8(s[group+0])) * (int32(l[i]&0x0F) | int32(h[i]&3)<<4) * int32(a[i])
				sum += int32(int8(s[group+2])) * (int32(l[i+32]&0x0F) | int32((h[i]>>2)&3)<<4) * int32(a[i+32])
				sum += int32(int8(s[group+4])) * (int32(l[i]>>4) | int32((h[i]>>4)&3)<<4) * int32(a[i+64])
				sum += int32(int8(s[group+6])) * (int32(l[i+32]>>4) | int32((h[i]>>6)&3)<<4) * int32(a[i+96])
			}
		}
		// The recentring, group by group: the weights were read shifted by 32.
		var shift int32
		for g, groupSum := range groupSums {
			shift += int32(int8(blockScales[g])) * int32(groupSum)
		}
		total += float32(sum-32*shift) * superScale * scales[b]
	}
	return total
}

// MatVecQ6_K computes y = W*x for Q6_K weights. x must already carry its Q8_K
// form.
//
// Q6_K exists in these files for the embedding tensors, which are normally read
// a row at a time and need no product at all. Gemma ties its logit head to its
// input embedding, so the same tensor is also a matrix of 262144 rows, and this
// is what reads it — three quarters of a gigabyte per token, which is what it
// costs to have no separate head.
func MatVecQ6_K(w []byte, b *Batch, outputs, inputs int, ys [][]float32) {
	if inputs%SuperBlock != 0 {
		panic("nn: Q6_K rows must be a multiple of the superblock size")
	}
	if b.QK == nil {
		panic("nn: a Q6_K product needs the activation in its Q8_K form")
	}
	InParallel(outputs, outputs*inputs*b.Size, func(start, end int) {
		matVecQ6_KRows(w, b, inputs, ys, start, end)
	})
}

// matVecQ6_KRows computes rows [start, end) on the caller's thread.
func matVecQ6_KRows(w []byte, b *Batch, inputs int, ys [][]float32, start, end int) {
	stride := inputs / SuperBlock * q6_kBlockBytes
	for r := start; r < end; r++ {
		row := w[r*stride : (r+1)*stride]
		for c := 0; c < b.Size; c++ {
			ys[c][r] = dotQ6_K(row,
				b.QK[c*b.Width:], b.BSums[c*b.Width/16:], b.KScales[c*b.Width/SuperBlock:], inputs)
		}
	}
}
