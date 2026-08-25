package nn

// The Q4_1 product and the Q4_1 row, against what ggml makes of the same bytes.
//
// The recording is ref/dump_quants.cpp run over Qwen3.8-27B, whose every
// ffn_down is Q4_1 — as is the ffn_down of the Q4_0 builds published for Qwen3,
// which is why qwen/quantized_test.go has to be pointed at a --pure file.

import (
	"math"
	"strconv"
	"testing"
)

// The tolerance is a thousandth where Q4_0's is a hundred-thousandth, and the
// reason is measured rather than allowed for: on this row it is ggml that is
// further from the answer, not this kernel. TestQ4_1IsNearerTheExactAnswer
// below is the assertion that says so, and it is the one to read first.
//
// What makes Q4_1 different from Q4_0 here is the minimum. A Q4_1 row's two
// terms — the products and the per-block minimums — are each far larger than
// the value they sum to, so the answer is a cancellation, and every rounding
// inside either term is amplified by the ratio between them. ggml accumulates
// the products in eight lanes and folds them at the end; this reads them in
// order. Neither is wrong, and on a row of seventeen thousand the two land two
// ten-thousandths apart.
func TestMatrixQ4_1(t *testing.T) {
	f := loadQuantFixture(t, "q4_1_matvec")

	m := Matrix{Data: f.Weights, Quant: Q4_1, Rows: f.Rows, Cols: f.Cols}
	x := NewBatch(f.Cols, 1)
	x.Set(0, f.X)

	y := make([]float32, f.Rows)
	m.MatVec(x, y)

	compareFloats(t, "matrix q4_1", y, f.Y, 1e-3)
}

// The same product in float64, from the dequantized weights and the quantized
// activation, which is the answer both float32 kernels are approximating.
//
// This is what a tolerance loosened against a reference has to be paid for
// with. Measured on the fixture's sixty-four rows: this kernel is 1.7e-6 from
// the float64 answer and ggml's own recording is 2.4e-4 — a hundred and forty
// times further. The gap in the test above is ggml's accumulation, and a
// change here that made the two agree would be a change that made this engine
// less accurate.
func TestQ4_1IsNearerTheExactAnswer(t *testing.T) {
	f := loadQuantFixture(t, "q4_1_matvec")

	m := Matrix{Data: f.Weights, Quant: Q4_1, Rows: f.Rows, Cols: f.Cols}
	x := NewBatch(f.Cols, 1)
	x.Set(0, f.X)

	got := make([]float32, f.Rows)
	m.MatVec(x, got)

	// The activation as every kernel sees it: the Q8_0 quanta times their
	// block scale, not the floats they were made from.
	activation := make([]float64, f.Cols)
	for b := 0; b < f.Cols/QuantBlock; b++ {
		for j := 0; j < QuantBlock; j++ {
			at := b*QuantBlock + j
			activation[at] = float64(x.Q[at]) * float64(x.Scales[b])
		}
	}

	row := make([]float32, f.Cols)
	var ours, theirs float64
	for r := 0; r < f.Rows; r++ {
		m.Row(r, row)
		var exact float64
		for i := range row {
			exact += float64(row[i]) * activation[i]
		}
		ours = math.Max(ours, math.Abs(float64(got[r])-exact))
		theirs = math.Max(theirs, math.Abs(float64(f.Y[r])-exact))
	}
	t.Logf("worst gap to the float64 answer: this kernel %.3g, ggml %.3g", ours, theirs)
	if ours > theirs {
		t.Fatalf("this kernel is %.3g from the exact answer and ggml is %.3g: the loosened "+
			"tolerance above is no longer paid for", ours, theirs)
	}
}

// A batch of several columns, which is the loop the row is read once for.
// Every column is the same vector, so every answer is the same product: what
// this catches is an index that walks the batch where it should walk the row.
func TestMatrixQ4_1Batch(t *testing.T) {
	f := loadQuantFixture(t, "q4_1_matvec")

	const columns = 3
	m := Matrix{Data: f.Weights, Quant: Q4_1, Rows: f.Rows, Cols: f.Cols}
	x := NewBatch(f.Cols, columns)
	for c := 0; c < columns; c++ {
		x.Set(c, f.X)
	}
	ys := make([][]float32, columns)
	for c := range ys {
		ys[c] = make([]float32, f.Rows)
	}
	m.MatVecBatch(x, ys)

	for c := range ys {
		compareFloats(t, "matrix q4_1 column "+strconv.Itoa(c), ys[c], f.Y, 1e-3)
	}
}

func TestDequantizeQ4_1Row(t *testing.T) {
	f := loadQuantFixture(t, "q4_1_dequant")

	m := Matrix{Data: f.Weights, Quant: Q4_1, Rows: f.Rows, Cols: f.Cols}
	out := make([]float32, f.Cols)
	for r := 0; r < f.Rows; r++ {
		m.Row(r, out)
		compareFloats(t, "q4_1 row "+strconv.Itoa(r), out, f.Y[r*f.Cols:(r+1)*f.Cols], 1e-6)
	}
}
