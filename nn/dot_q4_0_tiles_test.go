package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The tiled path against the plain one, at widths the tile does not divide.
//
// A row wider than kTile is computed in stretches, and the last stretch is
// whatever is left. Gemma 4's 12B is where that stops being hypothetical: it
// projects over 3840 inputs and 15360, and neither is a multiple of the tile.
// Told to read a whole tile at the end of the row, the kernel reads past both
// the row and the activation, and the sum it returns has no bound at all.
//
// Reading past the end is only visible when there is something on the other
// side, so both operands here are longer than the product is told they are and
// the excess is filled: a freshly allocated buffer would answer the overrun
// with zeros and the test would pass while the arithmetic was wrong.
func TestMatVecQ4_0PartialTiles(t *testing.T) {
	rng := rand.New(rand.NewSource(11))

	for _, inputs := range []int{2048, 2080, 3840, 4096, 6144, 15360} {
		for _, size := range []int{1, 2, 6, 8, 13} {
			const rows = 5
			blocks := inputs / QuantBlock
			rowBytes := blocks * q4_0BlockBytes

			// One tile of slack after the last row, so that an overrun there
			// meets numbers rather than the end of the allocation.
			w := make([]byte, rows*rowBytes+kTile/QuantBlock*q4_0BlockBytes)
			for i := range w {
				w[i] = byte(rng.Intn(256))
			}
			// A random fp16 can be NaN or enormous; keep the scales ordinary.
			for b := 0; b < len(w)/q4_0BlockBytes; b++ {
				w[b*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
			}

			// The activation is quantized a tile wider than it is used, and
			// then presented as a batch of the narrower width — which is what
			// a reused scratch buffer looks like.
			full := NewBatch(inputs+kTile, size)
			for c := 0; c < size; c++ {
				for i := range full.F[c] {
					full.F[c][i] = rng.Float32()*2 - 1
				}
			}
			full.Quantize()
			b := &Batch{
				Size: size, Stride: full.Stride, Width: inputs,
				F: make([][]float32, size), Q: full.Q,
				Scales: full.Scales, Corr: full.Corr,
			}
			for c := range b.F {
				b.F[c] = full.F[c][:inputs]
			}

			ys := make([][]float32, size)
			for c := range ys {
				ys[c] = make([]float32, rows)
			}
			matVecQ4_0Rows(w, b, inputs, ys, 0, rows)

			for c := 0; c < size; c++ {
				for r := 0; r < rows; r++ {
					var state [9]float32
					dotQ4_0Go(w[r*rowBytes:], b, 0, c, inputs, state[:])
					want, got := fold(state[:], state[8]), ys[c][r]
					if gap := math.Abs(float64(got - want)); gap > 1e-3*math.Abs(float64(want))+1e-4 {
						t.Fatalf("inputs=%d size=%d row %d column %d: tiled %.6f, plain %.6f",
							inputs, size, r, c, got, want)
					}
				}
			}
		}
	}
}
