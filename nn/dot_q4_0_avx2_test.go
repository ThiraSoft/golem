//go:build amd64

package nn

import (
	"math"
	"math/rand"
	"testing"
)

// The kernels against the portable form, on random blocks. They agree to
// rounding: the same products, the assembly eight lanes at a time and the
// portable form one block at a time.
func TestDotQ4_0AVX2(t *testing.T) {
	if !hasAVX2() {
		t.Skip("no AVX2 on this machine")
	}
	rng := rand.New(rand.NewSource(7))

	for _, n := range []int{32, 64, 256, 1536, 4096} {
		for _, size := range []int{1, 3, 4, 7, 8, 11, 16} {
			blocks := n / QuantBlock
			w := make([]byte, blocks*q4_0BlockBytes)
			for i := range w {
				w[i] = byte(rng.Intn(256))
			}
			// Keep the scales to sane magnitudes: random fp16 bits can be NaN.
			for b := 0; b < blocks; b++ {
				w[b*q4_0BlockBytes] = byte(rng.Intn(256))
				w[b*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
			}
			batch := NewBatch(n, size)
			for c := 0; c < size; c++ {
				for i := range batch.F[c] {
					batch.F[c][i] = rng.Float32()*2 - 1
				}
			}
			batch.Quantize()

			portable := func(column int) float32 {
				var state [9]float32
				dotQ4_0Go(w, batch, 0, column, n, state[:])
				return fold(state[:], state[8])
			}

			for c := 0; c < size; c++ {
				var state [9]float32
				dotQ4_0(w, batch, 0, c, n, state[:], Begin|Finish)
				got, want := state[0], portable(c)
				if gap := math.Abs(float64(got - want)); gap > 1e-3*math.Abs(float64(want))+1e-6 {
					t.Fatalf("n=%d size=%d column %d: assembly %.8f, portable %.8f", n, size, c, got, want)
				}
			}

			for c := 0; c+4 <= size; c += 4 {
				var state [36]float32
				dotQ4_0x4(w, batch, 0, c, n, state[:], Begin|Finish)
				for j := 0; j < 4; j++ {
					got, want := state[j], portable(c+j)
					if gap := math.Abs(float64(got - want)); gap > 1e-3*math.Abs(float64(want))+1e-6 {
						t.Fatalf("n=%d size=%d four columns at %d, %d: %.8f against %.8f",
							n, size, c, j, got, want)
					}
				}
			}

			for c := 0; c+8 <= size; c += 8 {
				var state [72]float32
				dotQ4_0x8(w, batch, 0, c, n, state[:], Begin|Finish)
				for j := 0; j < 8; j++ {
					got, want := state[j], portable(c+j)
					if gap := math.Abs(float64(got - want)); gap > 1e-3*math.Abs(float64(want))+1e-6 {
						t.Fatalf("n=%d size=%d eight columns at %d, %d: %.8f against %.8f",
							n, size, c, j, got, want)
					}
				}
			}

			// A row cut into stretches of input has to give exactly what the
			// same row gives in one call: the lanes are carried, not folded and
			// added.
			if n >= 2*QuantBlock {
				half := n / 2
				if half%QuantBlock == 0 {
					var wide [72]float32
					for c := 0; c+8 <= size; c += 8 {
						dotQ4_0x8(w, batch, 0, c, half, wide[:], Begin)
						dotQ4_0x8(w[half/QuantBlock*q4_0BlockBytes:], batch, half/QuantBlock, c, n-half, wide[:], Finish)
						var whole [72]float32
						dotQ4_0x8(w, batch, 0, c, n, whole[:], Begin|Finish)
						for j := 0; j < 8; j++ {
							if wide[j] != whole[j] {
								t.Fatalf("n=%d size=%d column %d: in two stretches %v, in one %v",
									n, size, c+j, wide[j], whole[j])
							}
						}
					}

					var state [36]float32
					for c := 0; c+4 <= size; c += 4 {
						dotQ4_0x4(w, batch, 0, c, half, state[:], Begin)
						dotQ4_0x4(w[half/QuantBlock*q4_0BlockBytes:], batch, half/QuantBlock, c, n-half, state[:], Finish)
						var whole [36]float32
						dotQ4_0x4(w, batch, 0, c, n, whole[:], Begin|Finish)
						for j := 0; j < 4; j++ {
							if state[j] != whole[j] {
								t.Fatalf("n=%d size=%d column %d: in two stretches %v, in one %v",
									n, size, c+j, state[j], whole[j])
							}
						}
					}
				}
			}
		}
	}
}
