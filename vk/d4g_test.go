package vk

// The D4G kernel against the CPU one, on weights a real quantizer produced.
//
// A synthetic matrix would not do: the twelve-bit codes are packed two to three
// bytes and the row is split into two planes, so a shader that reads a nibble
// from the wrong half, or a step from the wrong parity, still agrees with a
// matrix whose codes happen to be small. These weights carry the whole shell.

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/nn"
)

func d4gMatrix(tb testing.TB, rows, cols int) ([]byte, []float32) {
	tb.Helper()
	r := rand.New(rand.NewSource(17))
	w := make([]float32, rows*cols)
	for i := range w {
		// A few large coefficients among many small ones, which is the shape a
		// weight matrix has and a uniform draw does not.
		w[i] = float32(r.NormFloat64()) * 0.02 * float32(1+i%13) / 7
	}
	q := make([]float32, cols)
	for j := range q {
		q[j] = 1
		if r.Intn(2) == 0 {
			q[j] = -1
		}
	}
	data := compress.EncodeD4G(w, rows, cols, q, compress.D4Params{
		Beta: 2, ScaleBlock: 64, HadGroup: 128, SearchScale: true})
	return data, q
}

func TestD4GMatVecMatchesCPU(t *testing.T) {
	const rows, cols = 512, 1024
	d := open(t)
	defer d.Close()

	data, q := d4gMatrix(t, rows, cols)
	m := nn.Matrix{Data: data, Quant: nn.D4G, Rows: rows, Cols: cols}

	// The activation, through the preparation both sides assume.
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.37)) * float32(1+i%17) * 0.11
	}
	pre := make([]float32, cols)
	for j := range pre {
		pre[j] = 1 / q[j]
	}
	nn.PrepareD4G(x, pre, 128)

	b := nn.NewBatch(cols, 1)
	copy(b.F[0], x)
	want := make([]float32, rows)
	m.MatVec(b, want)

	gpu, err := NewD4GMatrix(d, data, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	got := make([]float32, rows)
	if err := gpu.MatVec(x, got); err != nil {
		t.Fatal(err)
	}

	var worst float64
	var where int
	var scale float64
	for _, v := range want {
		scale = math.Max(scale, math.Abs(float64(v)))
	}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d/scale > worst {
			worst, where = d/scale, i
		}
	}
	// Both sides sum the same integers times the same fp16 steps; only the
	// order of the additions differs, so the gap is float32 rounding over a
	// thousand terms and nothing else.
	if worst > 1e-5 {
		t.Fatalf("row %d diverges: CPU %v, GPU %v (relative %g)", where, want[where], got[where], worst)
	}
	t.Logf("%d rows of %d, worst gap %g of the largest output", rows, cols, worst)
}

func make2(n int) []float32 { return make([]float32, n) }

func BenchmarkD4GMatVecGPU(b *testing.B) {
	for _, c := range []struct{ rows, cols int }{
		{16384, 4096}, {32768, 4096},
	} {
		for _, noTable := range []bool{false, true} {
			name := fmt.Sprintf("%dx%d", c.rows, c.cols)
			if noTable {
				name += "/no-table"
			}
			b.Run(name, func(b *testing.B) { benchD4G(b, c.rows, c.cols, noTable) })
		}
	}
}

func benchD4G(b *testing.B, rows, cols int, noTable bool) {
	d := open(b)
	defer d.Close()
	data, q := d4gMatrix(b, rows, cols)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.31))
	}
	pre := make([]float32, cols)
	for j := range pre {
		pre[j] = 1 / q[j]
	}
	nn.PrepareD4G(x, pre, 128)

	make := NewD4GMatrix
	if noTable {
		make = newD4GMatrixNoTable
	}
	gpu, err := make(d, data, rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer gpu.Close()
	out := make2(rows)
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := gpu.MatVec(x, out); err != nil {
			b.Fatal(err)
		}
	}
}
