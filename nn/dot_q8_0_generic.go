//go:build !amd64

package nn

// dotQ8_0 adds one row against one column of the batch, over the n inputs that
// begin at the given block, into a state of eight lanes.
func dotQ8_0(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if mode&Begin != 0 {
		clear(state)
	}
	dotQ8_0Go(w, b, block, column, n, state)
	if mode&Finish != 0 {
		state[0] = fold(state, state[8])
	}
}
