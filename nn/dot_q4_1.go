package nn

// The Q4_1 product.
//
// A Q4_1 block is 20 bytes: an fp16 scale, an fp16 minimum, then 32 weights
// packed two per byte the way Q4_0 packs them — the low nibble of byte j holds
// weight j and the high nibble weight j+16. The difference from Q4_0 is what
// the nibble means. Q4_0 stores the weight plus eight and recentres on the way
// out; Q4_1 stores it as it is, in 0..15, and carries a per-block minimum
// instead. A weight is d*n + m.
//
// That second term costs nothing, because the activation already carries what
// it needs. ggml pairs Q4_1 with a Q8_1 activation, whose block keeps the sum
// of its own quanta beside its scale; golem's Batch keeps the same quantity in
// Corr, scaled by eight for Q4_0's recentring. So the minimum's contribution is
// m * Corr/8, and division by eight is exact.

//
// This is where the published Qwen3 Q4_0 builds keep ffn_down, and every
// ffn_down of Qwen3.8-27B.

import "encoding/binary"

// dotQ4_1Go is one row against one column of a batch, summed the way ggml sums
// it: the products in one running total, the minimums in a second, and the two
// added once at the end. That separation is not tidiness — ggml's kernel keeps
// the minimums in a scalar beside its vector accumulator, and folding them in
// block by block instead lands a quarter of a thousandth away.
//
// The two scales are multiplied together before they meet the integer sum, for
// the same reason: that is the order the kernel associates them in.
func dotQ4_1Go(w []byte, b *Batch, first, column, n int) float32 {
	var sum, mins float32
	for step := 0; step < n/QuantBlock; step++ {
		bytes := w[step*q4_1BlockBytes : (step+1)*q4_1BlockBytes]
		scale := halfToFloat(binary.LittleEndian.Uint16(bytes))
		minimum := halfToFloat(binary.LittleEndian.Uint16(bytes[2:]))
		nibbles := bytes[4:]
		index := (first+step)*b.Stride + column
		q := b.Q[index*QuantBlock : (index+1)*QuantBlock]

		var accumulator int32
		for j := 0; j < QuantBlock/2; j++ {
			byteValue := nibbles[j]
			accumulator += int32(byteValue&0x0F)*int32(q[j]) + int32(byteValue>>4)*int32(q[j+16])
		}
		// Corr is eight times the activation's scale times the sum of its
		// quanta, which is what Q4_0's recentring wants; Q4_1's minimum wants
		// the same product without the eight.
		sum += float32(accumulator) * (scale * b.Scales[index])
		mins += minimum * (b.Corr[index] / 8)
	}
	return sum + mins
}

// matVecQ4_1Rows computes rows [start, end) for every column of the batch.
//
// The row is the outer loop and the batch the inner one, for the reason
// nn/matrix.go gives: a row is read from memory once for the whole batch.
// There is no interleaved form and no wide kernel here — Q4_1 appears on one
// matrix of a block, where Q4_0 is on the other six.
func matVecQ4_1Rows(w []byte, b *Batch, inputs int, ys [][]float32, start, end int) {
	rowBytes := inputs / QuantBlock * q4_1BlockBytes
	for r := start; r < end; r++ {
		row := w[r*rowBytes:]
		for c := 0; c < b.Size; c++ {
			ys[c][r] = dotQ4_1Go(row, b, 0, c, inputs)
		}
	}
}

// dequantizeQ4_1Row expands one Q4_1 row, for the Row path.
func dequantizeQ4_1Row(w []byte, n int, out []float32) {
	if n%QuantBlock != 0 {
		panic("nn: Q4_1 rows must be a multiple of the block size")
	}
	for b := 0; b < n/QuantBlock; b++ {
		block := w[b*q4_1BlockBytes : (b+1)*q4_1BlockBytes]
		scale := halfToFloat(binary.LittleEndian.Uint16(block))
		minimum := halfToFloat(binary.LittleEndian.Uint16(block[2:]))
		nibbles := block[4:]
		dst := out[b*QuantBlock : (b+1)*QuantBlock]
		for j := 0; j < QuantBlock/2; j++ {
			byteValue := nibbles[j]
			dst[j] = float32(byteValue&0x0F)*scale + minimum
			dst[j+16] = float32(byteValue>>4)*scale + minimum
		}
	}
}
