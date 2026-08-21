//go:build !amd64

package nn

func dotPackedQ4_0(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	if mode&Begin != 0 {
		clear(state[:PackedRows])
	}
	dotPackedQ4_0Go(w, b, block, column, n, state)
}

func dotPackedQ4_0x4(w []byte, b *Batch, block, column, n int, state []float32, mode Mode) {
	for c := 0; c < 4; c++ {
		if mode&Begin != 0 {
			clear(state[c*PackedRows : (c+1)*PackedRows])
		}
		dotPackedQ4_0Go(w, b, block, column+c, n, state[c*PackedRows:])
	}
}
