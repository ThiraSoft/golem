//go:build amd64

package nn

//go:noescape
func dotQ4_0AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)

//go:noescape
func dotQ4_0x8AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)

//go:noescape
func dotQ4_0x4AVX2(w *byte, q *int8, scales, corr *float32, n, stride int, state *float32, mode int)

// dotQ4_0 adds one row against one column of the batch, over the n inputs that
// begin at the given block, into a state of eight lanes and one correction.
func dotQ4_0(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if avx2 && n%QuantBlock == 0 {
		at := block*b.Stride + column
		dotQ4_0AVX2(&w[0], &b.Q[at*QuantBlock], &b.Scales[at], &b.Corr[at], n, b.Stride, &state[0], int(mode))
		return
	}
	if mode&Begin != 0 {
		clear(state)
	}
	dotQ4_0Go(w, b, block, column, n, state)
	if mode&Finish != 0 {
		state[0] = fold(state, state[8])
	}
}

// dotQ4_0x4 does the same for four consecutive columns: the row is unpacked
// once for the four. It is what a batch that is not a multiple of eight
// finishes on. Its state is the four eight-lane sums followed by
// the four corrections.
func dotQ4_0x4(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if avx2 && n%QuantBlock == 0 {
		at := block*b.Stride + column
		dotQ4_0x4AVX2(&w[0], &b.Q[at*QuantBlock], &b.Scales[at], &b.Corr[at], n, b.Stride, &state[0], int(mode))
		return
	}
	if mode&Begin != 0 {
		clear(state)
	}
	dotQ4_0x4Go(w, b, block, column, n, state)
	if mode&Finish != 0 {
		for c := 0; c < 4; c++ {
			state[c] = fold(state[c*8:], state[32+c])
		}
	}
}

// dotQ4_0x8 does the same for eight consecutive columns. Its state is the
// eight eight-lane sums followed by the eight corrections.
func dotQ4_0x8(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if avx2 && n%QuantBlock == 0 {
		at := block*b.Stride + column
		dotQ4_0x8AVX2(&w[0], &b.Q[at*QuantBlock], &b.Scales[at], &b.Corr[at], n, b.Stride, &state[0], int(mode))
		return
	}
	if mode&Begin != 0 {
		clear(state)
	}
	dotQ4_0x8Go(w, b, block, column, n, state)
	if mode&Finish != 0 {
		for c := 0; c < 8; c++ {
			state[c] = fold(state[c*8:], state[64+c])
		}
	}
}
