package nn

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// The tiled product against the one-dot-a-column path, on shapes that leave a
// remainder in both directions: rows past the last group of four, columns past
// the last group of three, and a batch narrower than one tile.
func TestMatMulF16MatchesTheDotPath(t *testing.T) {
	if !TiledProducts() {
		t.Skip("no tiled kernels on this machine")
	}
	rng := rand.New(rand.NewSource(1))
	for _, shape := range []struct{ rows, cols, batch int }{
		{4, 32, 3}, {13, 64, 7}, {3, 32, 2}, {37, 768, 11}, {9, 3072, 1},
	} {
		halves := make([]uint16, shape.rows*shape.cols)
		for i := range halves {
			halves[i] = floatToHalf(float32(rng.NormFloat64() * 0.05))
		}
		data := unsafe.Slice((*byte)(unsafe.Pointer(&halves[0])), 2*len(halves))
		m := Matrix{Data: data, Quant: F16, Rows: shape.rows, Cols: shape.cols,
			Bias: make([]float32, shape.rows)}
		for i := range m.Bias {
			m.Bias[i] = float32(rng.NormFloat64())
		}
		b := NewBatch(shape.cols, shape.batch)
		x := make([]float32, 0, shape.batch*shape.cols)
		for c := range b.F {
			for i := range b.F[c] {
				b.F[c][i] = RoundHalf(float32(rng.NormFloat64()))
			}
			x = append(x, b.F[c]...)
		}
		got := make([][]float32, shape.batch)
		want := make([][]float32, shape.batch)
		for c := range got {
			got[c] = make([]float32, shape.rows)
			want[c] = make([]float32, shape.rows)
		}
		if !m.MatMulF16(x, got) {
			t.Fatal("MatMulF16 declined a shape it takes")
		}
		m.MatVecBatch(b, want)
		for c := range got {
			for r := range got[c] {
				// Another summation order and a fused multiply-add: the
				// same product to the precision a float32 sum of this
				// length carries.
				if d := math.Abs(float64(got[c][r] - want[c][r])); d > 1e-5*math.Sqrt(float64(shape.cols)) {
					t.Fatalf("%v: column %d row %d: %g, want %g", shape, c, r, got[c][r], want[c][r])
				}
			}
		}
	}
}

// GemmF32NT with strides wider than the rows, the way an attention reads heads
// out of a fused projection, against the plain double loop.
func TestGemmF32NTMatchesTheLoop(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, shape := range []struct{ n, rows, cols int }{
		{64, 4, 3}, {64, 9, 7}, {64, 1, 1}, {24, 13, 5}, {12, 5, 4},
	} {
		wStride, xStride, outStride := shape.n+40, shape.n+24, shape.cols+3
		w := make([]float32, shape.rows*wStride)
		x := make([]float32, shape.cols*xStride)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		out := make([]float32, shape.rows*outStride)
		GemmF32NT(w, wStride, x, xStride, shape.n, shape.rows, shape.cols, out, outStride)
		for r := 0; r < shape.rows; r++ {
			for c := 0; c < shape.cols; c++ {
				var want float64
				for k := 0; k < shape.n; k++ {
					want += float64(w[r*wStride+k]) * float64(x[c*xStride+k])
				}
				if d := math.Abs(float64(out[r*outStride+c]) - want); d > 1e-4 {
					t.Fatalf("%v: [%d][%d] = %g, want %g", shape, r, c, out[r*outStride+c], want)
				}
			}
		}
	}
}
