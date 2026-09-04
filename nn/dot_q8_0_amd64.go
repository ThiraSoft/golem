//go:build amd64

package nn

//go:noescape
func dotQ8_0AVX2(w *byte, q *int8, scales *float32, n, stride int, state *float32, mode int)

// dotQ8_0 adds one row against one column of the batch, over the n inputs that
// begin at the given block, into a state of eight lanes. Lane 8 is the
// correction Q4_0 keeps there and Q8_0 never writes, so fold serves both.
func dotQ8_0(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if avx2 && n%QuantBlock == 0 {
		at := block*b.Stride + column
		dotQ8_0AVX2(&w[0], &b.Q[at*QuantBlock], &b.Scales[at], n, b.Stride, &state[0], int(mode))
		return
	}
	if mode&Begin != 0 {
		clear(state)
	}
	dotQ8_0Go(w, b, block, column, n, state)
	if mode&Finish != 0 {
		state[0] = fold(state, state[8])
	}
}
