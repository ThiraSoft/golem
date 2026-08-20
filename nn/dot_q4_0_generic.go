//go:build !amd64

package nn

func dotQ4_0(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if mode&Begin != 0 {
		clear(state)
	}
	dotQ4_0Go(w, b, block, column, n, state)
	if mode&Finish != 0 {
		state[0] = fold(state, state[8])
	}
}

func dotQ4_0x4(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
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
