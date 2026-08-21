//go:build amd64

package nn

//go:noescape
func dotPackedQ4_0x4AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)

//go:noescape
func dotPackedQ4_0AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)

// dotPackedQ4_0 adds one group of eight rows against one column of the batch,
// over the n inputs beginning at the given block, into eight lanes — a lane to
// a row.
func dotPackedQ4_0(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if avx2 && n%QuantBlock == 0 {
		at := block*b.Stride + column
		dotPackedQ4_0AVX2(&w[0], &b.Q[at*QuantBlock], &b.Scales[at], &b.Corr[at], n, b.Stride, &state[0], int(mode))
		return
	}
	if mode&Begin != 0 {
		clear(state[:PackedRows])
	}
	dotPackedQ4_0Go(w, b, block, column, n, state)
}

// dotPackedQ4_0x4 does the same for four consecutive columns, which is what
// the group is read once for. Its state is the four sets of eight lanes.
func dotPackedQ4_0x4(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if avx2 && n%QuantBlock == 0 {
		at := block*b.Stride + column
		dotPackedQ4_0x4AVX2(&w[0], &b.Q[at*QuantBlock], &b.Scales[at], &b.Corr[at], n, b.Stride, &state[0], int(mode))
		return
	}
	for c := 0; c < 4; c++ {
		if mode&Begin != 0 {
			clear(state[c*PackedRows : (c+1)*PackedRows])
		}
		dotPackedQ4_0Go(w, b, block, column+c, n, state[c*PackedRows:])
	}
}
