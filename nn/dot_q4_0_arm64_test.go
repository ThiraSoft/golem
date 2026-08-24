//go:build arm64

package nn

import (
	"math/rand"
	"testing"
)

// The NEON kernel against the portable form, on random blocks.
//
// They agree exactly, not closely, and the test says so. The assembly does the
// thirty-two products of a block in integers, where the sum cannot overflow, so
// SDOT's four lanes and the portable form's one accumulator reach the same
// number; everything after that — the fp16 scale, the two multiplies, the
// running total — is the same Go code in the same order. A gap of one unit in
// the last place here means a real difference, not a rounding, which is a much
// sharper instrument than a tolerance would be.
func TestDotQ4_0NEONMatchesTheGoFormExactly(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(7))

	for _, n := range []int{32, 64, 256, 1536, 2048, 4096} {
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

			for c := 0; c < size; c++ {
				var want [9]float32
				dotQ4_0Go(w, batch, 0, c, n, want[:])

				var got [9]float32
				dotQ4_0NEON(w, batch, 0, c, n, got[:])

				if got != want {
					t.Fatalf("n=%d size=%d column=%d: NEON %v, portable %v", n, size, c, got, want)
				}
			}
		}
	}
}

// More blocks than one trip into the assembly covers, so the slicing across
// calls is exercised rather than assumed. 4096 inputs is 128 blocks against a
// sumsPerCall of 64, and 2080 is 65 — one block into a second trip, which is
// where an off-by-one in the batching would show.
func TestDotQ4_0NEONCrossesTheCallBoundary(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(11))

	for _, blocks := range []int{sumsPerCall - 1, sumsPerCall, sumsPerCall + 1, 2 * sumsPerCall, 2*sumsPerCall + 3} {
		n := blocks * QuantBlock
		w := make([]byte, blocks*q4_0BlockBytes)
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		for b := 0; b < blocks; b++ {
			w[b*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
		}
		batch := NewBatch(n, 2)
		for c := 0; c < 2; c++ {
			for i := range batch.F[c] {
				batch.F[c][i] = rng.Float32()*2 - 1
			}
		}
		batch.Quantize()

		for c := 0; c < 2; c++ {
			var want, got [9]float32
			dotQ4_0Go(w, batch, 0, c, n, want[:])
			dotQ4_0NEON(w, batch, 0, c, n, got[:])
			if got != want {
				t.Fatalf("%d blocks, column %d: NEON %v, portable %v", blocks, c, got, want)
			}
		}
	}
}

// The kernel is also reached through the dispatch, in every mode and starting
// at a block other than zero — which is how a row cut into stretches uses it.
func TestDotQ4_0NEONThroughTheDispatch(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(13))

	const blocks, size = 96, 8
	const n = blocks * QuantBlock
	w := make([]byte, blocks*q4_0BlockBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	for b := 0; b < blocks; b++ {
		w[b*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
	}
	batch := NewBatch(n, size)
	for c := 0; c < size; c++ {
		for i := range batch.F[c] {
			batch.F[c][i] = rng.Float32()*2 - 1
		}
	}
	batch.Quantize()

	// A row taken whole, and the same row taken in two stretches that carry the
	// lanes across. Each is held against the portable form cut the same way:
	// splitting a row changes the order the block sums are added in, which
	// moves the last bit whichever code does it, so the whole and the split are
	// not each other's reference. The kernel's business is to match the
	// portable form at the same cut, and it does so exactly.
	half := n / 2
	halfBlocks := half / QuantBlock

	for c := 0; c < size; c++ {
		var whole [9]float32
		dotQ4_0(w, batch, 0, c, n, whole[:], Begin|Finish)

		var wantWhole [9]float32
		dotQ4_0Go(w, batch, 0, c, n, wantWhole[:])
		if want := fold(wantWhole[:], wantWhole[8]); whole[0] != want {
			t.Fatalf("column %d whole: %v, want %v", c, whole[0], want)
		}

		var split [9]float32
		dotQ4_0(w, batch, 0, c, half, split[:], Begin)
		dotQ4_0(w[halfBlocks*q4_0BlockBytes:], batch, halfBlocks, c, half, split[:], Finish)

		var wantSplit [9]float32
		dotQ4_0Go(w, batch, 0, c, half, wantSplit[:])
		dotQ4_0Go(w[halfBlocks*q4_0BlockBytes:], batch, halfBlocks, c, half, wantSplit[:])
		if want := fold(wantSplit[:], wantSplit[8]); split[0] != want {
			t.Fatalf("column %d split: %v, want %v", c, split[0], want)
		}
	}
}

// The x4 and x8 forms against the portable ones, in the state layout their
// callers read: the sums eight lanes apart, then the corrections.
func TestDotQ4_0NEONWideForms(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(17))

	const blocks = 40
	const n = blocks * QuantBlock
	w := make([]byte, blocks*q4_0BlockBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	for b := 0; b < blocks; b++ {
		w[b*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
	}
	batch := NewBatch(n, 8)
	for c := 0; c < 8; c++ {
		for i := range batch.F[c] {
			batch.F[c][i] = rng.Float32()*2 - 1
		}
	}
	batch.Quantize()

	var got4, want4 [36]float32
	dotQ4_0x4(w, batch, 0, 0, n, got4[:], Begin)
	dotQ4_0x4Go(w, batch, 0, 0, n, want4[:])
	if got4 != want4 {
		t.Errorf("x4: %v, want %v", got4, want4)
	}

	var got8, want8 [72]float32
	dotQ4_0x8(w, batch, 0, 0, n, got8[:], Begin)
	dotQ4_0x8Go(w, batch, 0, 0, n, want8[:])
	if got8 != want8 {
		t.Errorf("x8: %v, want %v", got8, want8)
	}
}
