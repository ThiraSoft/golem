//go:build arm64

package nn

import "encoding/binary"

//go:noescape
func packedQ4_0SumsNEON(w *byte, q *int8, blocks, qStride int, sums *int32)

// dotPackedQ4_0NEON is dotPackedQ4_0Go with the products done by SDOT: four
// rows a lane, eight inputs a pass. The float scales stay in Go, as they do in
// the other kernels here, and the integer sums are exact either way — so this
// and the portable form agree bit for bit.
func dotPackedQ4_0NEON(w []byte, b *Batch, block, column, n int, state []float32) {
	// Four sums a block rather than one, so the buffer covers a quarter as
	// many blocks for the same quarter-kilobyte.
	var sums [sumsPerCall * PackedRows]int32

	blocks := n / QuantBlock
	qStride := b.Stride * QuantBlock
	for done := 0; done < blocks; done += sumsPerCall {
		batch := min(sumsPerCall, blocks-done)
		at := (block+done)*b.Stride + column
		packedQ4_0SumsNEON(&w[done*packedBlockBytes], &b.Q[at*QuantBlock], batch, qStride, &sums[0])

		for i := 0; i < batch; i++ {
			step := done + i
			group := w[step*packedBlockBytes : (step+1)*packedBlockBytes]
			index := (block+step)*b.Stride + column
			activation, correction := b.Scales[index], b.Corr[index]
			for r := 0; r < PackedRows; r++ {
				scale := halfToFloat(binary.LittleEndian.Uint16(group[r*2:]))
				state[r] += float32(sums[i*PackedRows+r]) * (scale * activation)
				state[r] -= scale * correction
			}
		}
	}
}

func dotPackedQ4_0(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if mode&Begin != 0 {
		clear(state[:PackedRows])
	}
	if dotprod && n%QuantBlock == 0 {
		dotPackedQ4_0NEON(w, b, block, column, n, state)
	} else {
		dotPackedQ4_0Go(w, b, block, column, n, state)
	}
}

func dotPackedQ4_0x4(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	for c := 0; c < 4; c++ {
		if mode&Begin != 0 {
			clear(state[c*PackedRows : (c+1)*PackedRows])
		}
		if dotprod && n%QuantBlock == 0 {
			dotPackedQ4_0NEON(w, b, block, column+c, n, state[c*PackedRows:])
		} else {
			dotPackedQ4_0Go(w, b, block, column+c, n, state[c*PackedRows:])
		}
	}
}
