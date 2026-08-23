//go:build arm64

package nn

import "encoding/binary"

//go:noescape
func q4_0SumsNEON(w *byte, q *int8, blocks, qStride int, sums *int32)

// sumsPerCall is how many blocks one trip into the assembly covers. The sums
// come back through the stack rather than through an allocation, so the buffer
// is fixed; sixty-four of them is a quarter of a kilobyte, which stays in the
// first-level cache while the Go half reads it straight back.
const sumsPerCall = 64

// dotQ4_0NEON is dotQ4_0Go with the thirty-two products of each block done by
// SDOT instead of by hand. Everything after the sum — the fp16 scale, the two
// multiplies, the running total — is the portable code unchanged, in the same
// order, so the two agree bit for bit rather than closely.
func dotQ4_0NEON(w []byte, b *Batch, first, column, n int, state []float32) {
	var sums [sumsPerCall]int32
	var sum, correction float32

	blocks := n / QuantBlock
	qStride := b.Stride * QuantBlock
	for done := 0; done < blocks; done += sumsPerCall {
		batch := min(sumsPerCall, blocks-done)
		at := (first+done)*b.Stride + column
		q4_0SumsNEON(&w[done*q4_0BlockBytes], &b.Q[at*QuantBlock], batch, qStride, &sums[0])

		for i := 0; i < batch; i++ {
			step := done + i
			scale := halfToFloat(binary.LittleEndian.Uint16(w[step*q4_0BlockBytes:]))
			index := (first+step)*b.Stride + column
			sum += float32(sums[i]) * scale * b.Scales[index]
			correction += scale * b.Corr[index]
		}
	}
	state[0] += sum
	state[8] += correction
}

// dotQ4_0 adds one row against one column of the batch, over the n inputs that
// begin at the given block, into a state of eight lanes and one correction.
//
// Only the first lane is ever written here, as in the portable form: the fold
// that follows adds all eight, and the seven zeroes cost nothing.
func dotQ4_0(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if mode&Begin != 0 {
		clear(state)
	}
	if dotprod && n%QuantBlock == 0 {
		dotQ4_0NEON(w, b, block, column, n, state)
	} else {
		dotQ4_0Go(w, b, block, column, n, state)
	}
	if mode&Finish != 0 {
		state[0] = fold(state, state[8])
	}
}

// dotQ4_0x4 does the same for four consecutive columns. Its state is the four
// eight-lane sums followed by the four corrections.
//
// The row is not unpacked once for the four here, as the AVX2 kernel does: the
// nibbles are read again per column. Whether that is worth a wider kernel is a
// question about a machine this was not written on, so it is left alone.
func dotQ4_0x4(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if mode&Begin != 0 {
		clear(state)
	}
	var one [9]float32
	for c := 0; c < 4; c++ {
		one[0], one[8] = 0, 0
		if dotprod && n%QuantBlock == 0 {
			dotQ4_0NEON(w, b, block, column+c, n, one[:])
		} else {
			dotQ4_0Go(w, b, block, column+c, n, one[:])
		}
		state[c*8] += one[0]
		state[32+c] += one[8]
	}
	if mode&Finish != 0 {
		for c := 0; c < 4; c++ {
			state[c] = fold(state[c*8:], state[32+c])
		}
	}
}

// dotQ4_0x8 does the same for eight consecutive columns. Its state is the eight
// eight-lane sums followed by the eight corrections.
func dotQ4_0x8(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if mode&Begin != 0 {
		clear(state)
	}
	var one [9]float32
	for c := 0; c < 8; c++ {
		one[0], one[8] = 0, 0
		if dotprod && n%QuantBlock == 0 {
			dotQ4_0NEON(w, b, block, column+c, n, one[:])
		} else {
			dotQ4_0Go(w, b, block, column+c, n, one[:])
		}
		state[c*8] += one[0]
		state[64+c] += one[8]
	}
	if mode&Finish != 0 {
		for c := 0; c < 8; c++ {
			state[c] = fold(state[c*8:], state[64+c])
		}
	}
}
