package vk

// The Q2_K products against the processor's own reader of the same bytes, on a
// real tensor.
//
// vk/split_q2k_test.go already holds the packing to nn's dequantiser byte for
// byte, and nn's is pinned to ggml's own output — but both of those would pass
// on a shader that never ran. This one is the other half: the kernels, at every
// width they are built for, on a checkpoint a quantizer wrote.
//
// Q2_K is where a real tensor matters most. Two bits leave four magnitudes, so
// a shader that read the pairs in the wrong order still produces weights in the
// right range on every row, and a scale and a minimum that were swapped —
// they are two nibbles of one byte — answer something plausible everywhere.

import (
	"math"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// TestVulkanQ2KMatVecMatchesCPU is the mat-vec at every width the door builds.
//
// Every width and not only one: the widths are separate binaries, and the
// column offset a wide one applies to its activation and its output is the part
// that is easy to get wrong and impossible to see — a kernel that read column
// zero for all sixteen answers the right shape.
func TestVulkanQ2KMatVecMatchesCPU(t *testing.T) {
	g, m := aQ2_K(t)
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

// TestVulkanQ2KTiledMatchesCPU is the same weights through the cooperative
// tiled product, which is what a prompt pass reaches and a token never does.
//
// It stages a weight as fp16 rather than dotting it as an integer, so it is a
// different reading of the same bytes and not the same code at another width —
// and for Q2_K it is the only reading that does not start from a nibble pair.
func TestVulkanQ2KTiledMatchesCPU(t *testing.T) {
	g, m := aQ2_K(t)
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

			mm, err := NewMatMulQuant(d, m.Data, m.Rows, m.Cols, columns, true, nn.Q2_K)
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
