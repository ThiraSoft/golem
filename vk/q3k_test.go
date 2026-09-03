package vk

// The Q3_K products against the processor's own reader of the same bytes, on a
// real tensor.
//
// vk/split_q3k_test.go already holds the packing to nn's dequantiser byte for
// byte, and that test would pass on a shader that never ran. This one is the
// other half: the kernels, at every width they are built for, on a checkpoint
// a quantizer wrote rather than on random bytes.
//
// A real tensor matters more here than for any other format in this repository.
// Q3_K's sixteen scales come out of twelve bytes in an order that is ggml's
// alone, and a random matrix's scales are uniform over six bits — so a reader
// that took the last four groups of a superblock from the wrong two bits would
// still agree with random weights on most rows. Two of the three K-quant
// readers in this repository shipped with exactly that fault.

import (
	"math"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
)

// TestVulkanQ3KMatVecMatchesCPU is the mat-vec at every width the door builds.
//
// Every width and not only one: the widths are separate binaries, and the
// column offset a wide one applies to its activation and its output is the part
// that is easy to get wrong and impossible to see — a kernel that read column
// zero for all sixteen answers the right shape.
func TestVulkanQ3KMatVecMatchesCPU(t *testing.T) {
	g, m := aQ3_K(t)
	defer g.Close()
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	for _, columns := range []int{1, 2, 4, 8, 16} {
		t.Run(itoa(columns), func(t *testing.T) {
			batch := columnsOf(m.Cols, columns)
			// Against the Q8_0 form of the activation, which is what this
			// kernel reads. Comparing to the float product instead would be
			// comparing two different products, and the gap would be that
			// rounding rather than anything under test.
			want := kQuantAgainstQ8_0(m, batch, columns)

			q, scales := q80Batch(batch, m.Cols, columns)
			got := quantProductWide(t, d, m, q, scales, columns)

			var worst, scale float64
			var atCol, atRow int
			for c := range got {
				for i := range got[c] {
					scale = math.Max(scale, math.Abs(float64(want[c][i])))
					if gap := math.Abs(float64(got[c][i] - want[c][i])); gap > worst {
						worst, atCol, atRow = gap, c, i
					}
				}
			}
			if worst > 1e-3*scale {
				t.Fatalf("column %d row %d: %v against the processor's %v, %g of a peak of %g",
					atCol, atRow, got[atCol][atRow], want[atCol][atRow], worst, scale)
			}
			t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, columns, worst, scale)
		})
	}
}

// TestVulkanQ3KTiledMatchesCPU is the same weights through the cooperative
// tiled product, which is what a prompt pass reaches and a token never does.
//
// It stages a weight as fp16 rather than dotting it as an integer, so it is a
// different reading of the same bytes and not the same code at another width —
// and for Q3_K it is the only reading that does not start from a nibble pair.
func TestVulkanQ3KTiledMatchesCPU(t *testing.T) {
	g, m := aQ3_K(t)
	defer g.Close()
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("no cooperative matrices on this device")
	}

	for _, columns := range []int{32, 64, 128, 256, 512} {
		t.Run(itoa(columns), func(t *testing.T) {
			batch := columnsOf(m.Cols, columns)
			want := kQuantAgainstQ8_0(m, batch, columns)

			mm, err := NewMatMulQuant(d, m.Data, m.Rows, m.Cols, columns, true, nn.Q3_K)
			if err != nil {
				t.Fatal(err)
			}
			defer mm.Close()
			for c := 0; c < columns; c++ {
				if err := mm.SetColumn(c, oneColumn(batch, c)); err != nil {
					t.Fatal(err)
				}
			}
			got := make([][]float32, columns)
			for c := range got {
				got[c] = make([]float32, m.Rows)
			}
			if err := mm.Run(got); err != nil {
				t.Fatal(err)
			}

			var worst, scale float64
			var atCol, atRow int
			for c := range got {
				for i := range got[c] {
					scale = math.Max(scale, math.Abs(float64(want[c][i])))
					if gap := math.Abs(float64(got[c][i] - want[c][i])); gap > worst {
						worst, atCol, atRow = gap, c, i
					}
				}
			}
			// A tenth of a per cent is the mat-vec's bar. The tiled path stages
			// each weight as an fp16 and accumulates in float32, which is one
			// more rounding a weight, so it is held to twice that and no more.
			if worst > 2e-3*scale {
				t.Fatalf("column %d row %d: %v against the processor's %v, %g of a peak of %g",
					atCol, atRow, got[atCol][atRow], want[atCol][atRow], worst, scale)
			}
			t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, columns, worst, scale)
		})
	}
}

// q80Batch is a whole batch in the form the kernels read: the columns laid out
// one after another, each the Q8_0 magnitudes of its own row of blocks, and
// the scales and sums of every column after them in the same order. q80Column
// is one column of it; this is the wide dispatch's whole buffer.
func q80Batch(batch *nn.Batch, cols, columns int) ([]uint32, []float32) {
	nb := cols / nn.QuantBlock
	q := make([]uint32, columns*nb*8)
	scales := make([]float32, columns*2*nb)
	for c := 0; c < columns; c++ {
		cq, cs, _ := q80Column(batch.F[c])
		copy(q[c*nb*8:], cq)
		copy(scales[c*2*nb:], cs)
	}
	return q, scales
}

// quantProductWide is quantProductOnCard at a width: the same door and the same
// four buffers, dispatched through the binary built for that many columns, and
// read back column by column.
func quantProductWide(tb testing.TB, d *Device, m nn.Matrix, q []uint32, scales []float32, columns int) [][]float32 {
	tb.Helper()
	layout, err := quantLayout(m.Quant, m.Data, m.Rows, m.Cols)
	if err != nil {
		tb.Fatal(err)
	}
	products := newQuantProducts(d, d.Coopmat())
	defer products.Close()
	pipe, err := products.get(m.Quant)
	if err != nil {
		tb.Fatal(err)
	}

	w, err := d.Upload(layout)
	if err != nil {
		tb.Fatal(err)
	}
	defer w.Close()
	aq, err := d.Upload(unsafe.Slice((*byte)(unsafe.Pointer(&q[0])), len(q)*4))
	if err != nil {
		tb.Fatal(err)
	}
	defer aq.Close()
	as, err := d.Upload(asBytes(scales))
	if err != nil {
		tb.Fatal(err)
	}
	defer as.Close()
	y, err := d.Readback(m.Rows*columns*4, bufferUsageStorage)
	if err != nil {
		tb.Fatal(err)
	}
	defer y.Close()

	set, err := pipe.NewSet([]*Buffer{w, aq, as, y})
	if err != nil {
		tb.Fatal(err)
	}
	defer set.Close()
	push := moePush{dim: uint32(m.Rows), ffn: uint32(m.Cols), used: 1, split: 1}
	if err := d.Submit(func(r *Recorder) {
		// One column is the pipeline's own binary rather than a Wide one, and
		// it is the width a generated token takes — so it is the width that
		// matters most here, not an edge of the table.
		if columns == 1 {
			r.Dispatch(set, groupsOf(m.Rows, matvecOuts), unsafe.Pointer(&push))
			return
		}
		r.DispatchWide(set, columns, groupsOf(m.Rows, matvecOuts), unsafe.Pointer(&push))
	}); err != nil {
		tb.Fatal(err)
	}
	out := make([][]float32, columns)
	floats := y.Floats()
	for c := range out {
		out[c] = make([]float32, m.Rows)
		copy(out[c], floats[c*m.Rows:(c+1)*m.Rows])
	}
	return out
}
