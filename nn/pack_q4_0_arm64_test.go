//go:build arm64

package nn

import (
	"math/rand"
	"testing"
)

func packedFixture(t testing.TB, rng *rand.Rand, cols, size int) ([]byte, *Batch) {
	t.Helper()
	blocks := cols / QuantBlock
	w := make([]byte, PackedRows*blocks*q4_0BlockBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	for i := 0; i < PackedRows*blocks; i++ {
		w[i*q4_0BlockBytes] = byte(rng.Intn(256))
		w[i*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
	}
	packed := make([]byte, PackedQ4_0Bytes(PackedRows, cols))
	PackQ4_0(w, PackedRows, cols, packed)

	batch := NewBatch(cols, size)
	for c := 0; c < size; c++ {
		for i := range batch.F[c] {
			batch.F[c][i] = rng.Float32()*2 - 1
		}
	}
	batch.Quantize()
	return packed, batch
}

// The packed kernel against the portable form of the same layout, exactly.
//
// Both accumulate the products in integers and apply the float scales in the
// same order afterwards, so there is nothing between them to round differently.
func TestDotPackedQ4_0NEONMatchesTheGoFormExactly(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(31))

	for _, cols := range []int{32, 64, 256, 1536, sumsPerCall * QuantBlock, (sumsPerCall + 1) * QuantBlock} {
		packed, batch := packedFixture(t, rng, cols, 3)
		for c := 0; c < batch.Size; c++ {
			want := make([]float32, PackedRows)
			dotPackedQ4_0Go(packed, batch, 0, c, cols, want)

			got := make([]float32, PackedRows)
			dotPackedQ4_0NEON(packed, batch, 0, c, cols, got)

			for r := 0; r < PackedRows; r++ {
				if got[r] != want[r] {
					t.Fatalf("cols=%d column %d row %d: NEON %v, portable %v", cols, c, r, got[r], want[r])
				}
			}
		}
	}
}

// The rows must come out in order — lane r holding row r is the whole reason
// this layout was chosen, and a swizzle that put them anywhere else would still
// give plausible numbers. So the kernel is held against the file's own layout,
// row by row, rather than only against the packed portable form.
func TestDotPackedQ4_0NEONKeepsRowOrder(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(37))

	const cols = 512
	blocks := cols / QuantBlock
	w := make([]byte, PackedRows*blocks*q4_0BlockBytes)
	for i := range w {
		w[i] = byte(rng.Intn(256))
	}
	for i := 0; i < PackedRows*blocks; i++ {
		w[i*q4_0BlockBytes] = byte(rng.Intn(256))
		w[i*q4_0BlockBytes+1] = byte(0x30 + rng.Intn(4))
	}
	packed := make([]byte, PackedQ4_0Bytes(PackedRows, cols))
	PackQ4_0(w, PackedRows, cols, packed)

	batch := NewBatch(cols, 1)
	for i := range batch.F[0] {
		batch.F[0][i] = rng.Float32()*2 - 1
	}
	batch.Quantize()

	got := make([]float32, PackedRows)
	dotPackedQ4_0NEON(packed, batch, 0, 0, cols, got)

	for r := 0; r < PackedRows; r++ {
		var lanes [9]float32
		dotQ4_0Go(w[r*blocks*q4_0BlockBytes:], batch, 0, 0, cols, lanes[:])
		want := fold(lanes[:], lanes[8])
		// Across layouts the float order does differ, so this one is to
		// rounding — it is checking which row, not which bit.
		if gap := got[r] - want; gap > 1e-3*abs32(want)+1e-5 || -gap > 1e-3*abs32(want)+1e-5 {
			t.Fatalf("row %d: packed kernel %.8f, row layout %.8f", r, got[r], want)
		}
	}
}

// Through the dispatch, in the state layout matVecPackedQ4_0Rows reads.
func TestDotPackedQ4_0NEONThroughTheDispatch(t *testing.T) {
	if !dotprod {
		t.Skip("no FEAT_DotProd on this machine")
	}
	rng := rand.New(rand.NewSource(41))

	const cols = 256
	packed, batch := packedFixture(t, rng, cols, 4)

	got := make([]float32, 4*PackedRows)
	dotPackedQ4_0x4(packed, batch, 0, 0, cols, got, Begin)

	want := make([]float32, 4*PackedRows)
	for c := 0; c < 4; c++ {
		dotPackedQ4_0Go(packed, batch, 0, c, cols, want[c*PackedRows:])
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("x4 lane %d: NEON %v, portable %v", i, got[i], want[i])
		}
	}
}

func abs32(f float32) float32 {
	if f < 0 {
		return -f
	}
	return f
}
