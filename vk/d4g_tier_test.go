package vk

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/nn"
)

// Both code widths, and every pass width the kernels are built for, against the
// processor reading the same bytes.
//
// This is the API a pipeline uses — kernels once for the device, a matrix bound
// to the activation it reads and the output it writes — so it is the one worth
// holding to the CPU. A pass of several columns is where a wrong stride hides:
// with one column every offset is zero and a kernel that forgot the column
// entirely would pass.
func TestD4GTiersAndPassWidths(t *testing.T) {
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const rows, cols = 256, 512
	r := rand.New(rand.NewSource(17))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64()) * 0.02
	}
	q := compress.RandomSigns(cols, 23)
	pre := make([]float32, cols)
	for j := range q {
		pre[j] = 1 / q[j]
	}

	for _, bits := range []int{nn.D4Bits, nn.D4Bits16} {
		p := compress.D4Params{Beta: 2, ScaleBlock: 32, HadGroup: 128, Bits: bits, SearchScale: true}
		data := compress.EncodeD4G(w, rows, cols, q, p, nil)

		k, err := NewD4GKernels(d, bits)
		if err != nil {
			t.Fatalf("%d bits: %v", bits, err)
		}
		for _, columns := range D4GWidths {
			act, err := d.Host(cols*columns*4, bufferUsageStorage)
			if err != nil {
				t.Fatal(err)
			}
			out, err := d.Readback(rows*columns*4, bufferUsageStorage)
			if err != nil {
				t.Fatal(err)
			}
			m, err := NewD4GMatrixOn(k, data, rows, cols, act, out)
			if err != nil {
				t.Fatalf("%d bits, %d columns: %v", bits, columns, err)
			}

			// One activation a column, each prepared the way inference does.
			xs := make([][]float32, columns)
			for c := range xs {
				x := make([]float32, cols)
				for i := range x {
					x[i] = float32(r.NormFloat64())
				}
				xs[c] = append([]float32(nil), x...)
				nn.PrepareD4G(x, pre, 128)
				copy(act.Floats()[c*cols:], x)
			}
			push := m.Push(0)
			if err := m.Set(columns).Dispatch(m.Groups(), unsafe.Pointer(&push)); err != nil {
				t.Fatal(err)
			}

			// The processor, from the same bytes.
			cpu := nn.Matrix{Data: data, Rows: rows, Cols: cols}
			cpu.Quant = nn.D4G
			if bits == nn.D4Bits16 {
				cpu.Quant = nn.D4G16
			}
			b := nn.NewBatch(cols, columns)
			for c := range xs {
				x := append([]float32(nil), xs[c]...)
				nn.PrepareD4G(x, pre, 128)
				copy(b.F[c], x)
			}
			want := make([][]float32, columns)
			for c := range want {
				want[c] = make([]float32, rows)
			}
			cpu.MatVecBatch(b, want)

			got := out.Floats()
			worst, scale := 0.0, 0.0
			for c := 0; c < columns; c++ {
				for i := 0; i < rows; i++ {
					v := float64(want[c][i])
					if a := math.Abs(v); a > scale {
						scale = a
					}
					if e := math.Abs(float64(got[c*rows+i]) - v); e > worst {
						worst = e
					}
				}
			}
			t.Logf("%2d bits, %d columns: worst gap %.3g of the largest output", bits, columns, worst/scale)
			if worst > 1e-4*scale {
				t.Errorf("%d bits, %d columns: the card and the processor disagree by %g", bits, columns, worst/scale)
			}
			m.Close()
			out.Close()
			act.Close()
		}
		k.Close()
	}
}
