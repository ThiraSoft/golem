package nn

import (
	"math"
	"testing"
	"unsafe"
)

// An F16 matrix product against the same numbers widened by hand. What it pins
// is the plumbing — the row stride, the byte order and the dispatch — not the
// kernel, which DotF32Half already had and the cache already used.
func TestF16MatrixProduct(t *testing.T) {
	const rows, cols = 5, 64
	half := make([]uint16, rows*cols)
	wide := make([]float32, rows*cols)
	for i := range half {
		v := float32(i%17)*0.125 - 1
		half[i] = Half(v)
		wide[i] = halfToFloat(half[i])
	}
	data := unsafe.Slice((*byte)(unsafe.Pointer(&half[0])), len(half)*2)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(i%7)*0.25 - 0.75
	}

	m := Matrix{Data: data, Quant: F16, Rows: rows, Cols: cols}
	b := NewBatch(cols, 1)
	b.Set(0, x)
	got := make([]float32, rows)
	m.MatVec(b, got)

	for r := 0; r < rows; r++ {
		var want float32
		for c := 0; c < cols; c++ {
			want += wide[r*cols+c] * x[c]
		}
		if math.Abs(float64(got[r]-want)) > 1e-4 {
			t.Errorf("row %d: %v, wanted about %v", r, got[r], want)
		}
	}
}

// A row read back must be the halves widened, in the file's byte order.
func TestF16RowWidens(t *testing.T) {
	half := []uint16{Half(1), Half(-0.5), Half(0.25), Half(3)}
	data := unsafe.Slice((*byte)(unsafe.Pointer(&half[0])), len(half)*2)
	m := Matrix{Data: data, Quant: F16, Rows: 1, Cols: 4}
	out := make([]float32, 4)
	m.Row(0, out)
	for i, want := range []float32{1, -0.5, 0.25, 3} {
		if out[i] != want {
			t.Errorf("element %d: %v, wanted %v", i, out[i], want)
		}
	}
}
