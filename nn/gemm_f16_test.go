package nn

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// GemmF16 against the product in float64, over widths that leave ragged
// tiles and a matrix whose rows end on half a task.
func TestGemmF16MatchesFloat64(t *testing.T) {
	if !GemmF16Available() {
		t.Skip("no kernel on this machine")
	}
	rng := rand.New(rand.NewSource(3))
	for _, shape := range [][3]int{{48, 64, 1}, {32, 520, 7}, {96, 1024, 100}, {16, 8, 13}} {
		rows, width, cols := shape[0], shape[1], shape[2]
		w := make([]uint16, rows*width)
		for i := range w {
			w[i] = floatToHalf(float32(rng.NormFloat64()))
		}
		m := Matrix{Quant: F16, Rows: rows, Cols: width,
			Data: unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), len(w)*2)}
		p, ok := PackF16(m)
		if !ok {
			t.Fatal("PackF16 declined")
		}
		x := make([]float32, cols*width)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		bias := make([]float32, rows)
		for i := range bias {
			bias[i] = float32(i)
		}
		out := make([]float32, cols*rows)
		p.GemmF16(x, cols, out, bias, false)
		for c := 0; c < cols; c++ {
			for r := 0; r < rows; r++ {
				var want, mag float64
				for k := 0; k < width; k++ {
					v := float64(halfToFloat(w[r*width+k])) * float64(x[c*width+k])
					want += v
					mag += math.Abs(v)
				}
				want += float64(bias[r])
				if d := math.Abs(float64(out[c*rows+r]) - want); d > 1e-5*(mag+1) {
					t.Fatalf("%v: out[%d][%d] = %v, want %v", shape, c, r, out[c*rows+r], want)
				}
			}
		}
	}
}

// GemmPanelsF32 against float64, with ragged columns and strides wider than
// the rows.
func TestGemmPanelsF32MatchesFloat64(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	n, rows, cols := 37, 32, 11
	aStride, xStride, cStride := 40, 50, 45
	a := make([]float32, n*aStride)
	x := make([]float32, cols*xStride)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	c := make([]float32, cols*cStride)
	GemmPanelsF32(a, aStride, x, xStride, n, rows, cols, c, cStride)
	for j := 0; j < cols; j++ {
		for r := 0; r < rows; r++ {
			var want float64
			for k := 0; k < n; k++ {
				want += float64(a[k*aStride+r]) * float64(x[j*xStride+k])
			}
			if math.Abs(float64(c[j*cStride+r])-want) > 1e-4 {
				t.Fatalf("c[%d][%d] = %v, want %v", j, r, c[j*cStride+r], want)
			}
		}
	}
}
