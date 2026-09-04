package nn

// The Q8_0 product against a Q8_0 activation, in integers.
//
// Q8_0 is Q4_0 with nothing packed: 34 bytes to a block of thirty-two, an fp16
// scale then one signed byte a weight. Two things follow from that, and both
// make this kernel shorter than dot_q4_0's. There is no unpacking — the bytes
// are already the weights — and there is no recentring, because a Q8_0 weight
// is signed where a Q4_0 nibble is an unsigned number waiting to have eight
// taken off it. The correction term that Q4_0 carries through its whole state
// does not exist here.
//
// What it costs instead is a sign. VPMADDUBSW wants its first operand
// unsigned, so the row is split into |w| and sign(w): the magnitudes multiply
// the activation, and the sign is moved onto the activation first. Two
// instructions, once per block, against the sixteen bytes Q4_0 spends
// unpacking nibbles.
//
// Until this existed, a Q8_0 matrix went through a scalar loop over float
// activations — the only weight format in nn without a product written for it,
// and the reason a Q8_0 GGUF ran slower than the Q4_0 of the same model by
// more than its size accounts for.

import (
	"encoding/binary"
)

// dotQ8_0Go adds one row against one column of the batch, over the n inputs
// that begin at the given block, into a state of eight lanes. Lane 8 is the
// correction Q4_0 keeps there; Q8_0 has none and leaves it alone, so that fold
// serves both.
func dotQ8_0Go(w []byte, b *Batch, first, column, n int, state []float32) {
	var sum float32
	for step := 0; step < n/QuantBlock; step++ {
		bytes := w[step*q8_0BlockBytes : (step+1)*q8_0BlockBytes]
		scale := halfToFloat(binary.LittleEndian.Uint16(bytes))
		index := (first+step)*b.Stride + column
		q := b.Q[index*QuantBlock : (index+1)*QuantBlock]

		var accumulator int32
		for j := 0; j < QuantBlock; j++ {
			accumulator += int32(int8(bytes[2+j])) * int32(q[j])
		}
		sum += float32(accumulator) * scale * b.Scales[index]
	}
	state[0] += sum
}

// matVecQ8_0Rows computes ys = W * b for rows in [start, end) of a Q8_0 matrix.
func matVecQ8_0Rows(data []byte, b *Batch, cols int, ys [][]float32, start, end int) {
	rowBytes := cols / QuantBlock * q8_0BlockBytes
	var state [9]float32
	for c := 0; c < b.Size; c++ {
		for r := start; r < end; r++ {
			dotQ8_0(data[r*rowBytes:], b, 0, c, cols, state[:], Begin|Finish)
			ys[c][r] = state[0]
		}
	}
}
