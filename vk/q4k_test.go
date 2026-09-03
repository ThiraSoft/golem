package vk

// The Q4_K products against the processor's own reader of the same bytes.
//
// On a real tensor and not a synthetic one, for the reason vk/matmul_test.go
// and vk/q6k_test.go both give: a quantizer's scales carry every pattern the
// six-bit packing can take, and a shader that unpacked the last four sub-blocks
// of a superblock the wrong way round would still agree with a random matrix on
// most rows. Two of the three K-quant readers in this repository shipped with
// exactly that fault, and both survived because the two sides shared it.

import (
	"math"
	"os"
	"strings"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// aQ4_K opens one Q4_K matrix out of whichever K-quant checkpoint this machine
// has. Any Q4_K_M serves: the format does not know which engine reads it.
func aQ4_K(tb testing.TB) (*tensors.GGUF, nn.Matrix) {
	return aKQuant(tb, "Q4_K", nn.Q4_K)
}

// aQ6_K is the same for the six-bit tier, which in a Q4_K_M is where the
// quantizer puts a quarter of the attention and half the feed forward's down
// projection.
func aQ6_K(tb testing.TB) (*tensors.GGUF, nn.Matrix) {
	return aKQuant(tb, "Q6_K", nn.Q6_K)
}

// aQ3_K is the three-bit tier, which lives in a different checkpoint: a Q4_K_M
// holds none, and a Q3_K_S is Q3_K throughout.
func aQ3_K(tb testing.TB) (*tensors.GGUF, nn.Matrix) {
	return aKQuant(tb, "Q3_K", nn.Q3_K)
}

// aQ2_K is the two-bit tier, which lives in a Q2_K checkpoint and nowhere else.
func aQ2_K(tb testing.TB) (*tensors.GGUF, nn.Matrix) {
	return aKQuant(tb, "Q2_K", nn.Q2_K)
}

func aKQuant(tb testing.TB, dtype string, q nn.Quant) (*tensors.GGUF, nn.Matrix) {
	tb.Helper()
	for _, env := range []string{"GOLEM_MODEL_Q2K", "GOLEM_MODEL_Q3KM", "GOLEM_MODEL_Q4KM", "GOLEM_MODEL_QWEN_Q4KM", "GOLEM_MODEL_12B_Q4KM"} {
		path := os.Getenv(env)
		if path == "" {
			continue
		}
		g, err := tensors.OpenGGUF(path)
		if err != nil {
			continue
		}
		// Which role carries which tier is the quantizer's business and it
		// differs between blocks, so the tensor is chosen by the type it has
		// and not by the name it goes under.
		for name, t := range g.Tensors {
			if t.DType != dtype || len(t.Shape) != 2 || t.Shape[0] < 256 {
				continue
			}
			if !strings.HasPrefix(name, "blk.") {
				continue
			}
			return g, nn.Matrix{Data: t.Raw, Quant: q, Cols: t.Shape[0], Rows: t.Shape[1]}
		}
		g.Close()
	}
	tb.Skipf("set GOLEM_MODEL_Q2K, GOLEM_MODEL_Q3KM or GOLEM_MODEL_Q4KM to a checkpoint holding a %s tensor", dtype)
	return nil, nn.Matrix{}
}

// TestVulkanQ4KMatVecMatchesCPU is shaders/matvec_q4k.comp at every width it is
// built for.
//
// Every width and not only one: the widths are separate binaries, and the
// column offset a wide one applies to its activation and its output is the part
// that is easy to get wrong and impossible to see — a kernel that read column
// zero for all sixteen answers the right shape.
func TestVulkanQ4KMatVecMatchesCPU(t *testing.T) {
	g, m := aQ4_K(t)
	defer g.Close()
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const columns = 16
	batch := columnsOf(m.Cols, columns)
	want := make([][]float32, columns)
	for c := range want {
		want[c] = make([]float32, m.Rows)
	}
	m.MatVecBatch(batch, want)

	weights, err := d.Upload(splitQ4_K(m.Data, m.Rows, m.Cols))
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	x, err := d.Host(m.Cols*columns*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	xs := x.Floats()
	for c := 0; c < columns; c++ {
		copy(xs[c*m.Cols:], batch.F[c])
	}
	y, err := d.Readback(m.Rows*columns*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Close()

	pipe, err := d.NewPipeline(matvecQ4KSPIRV, 3, uint32(unsafe.Sizeof(matvecKPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	for _, w := range []struct {
		columns int
		spirv   []byte
	}{
		{2, matvecQ4K_2SPIRV}, {4, matvecQ4K_4SPIRV}, {8, matvecQ4K_8SPIRV}, {16, matvecQ4K_16SPIRV},
	} {
		if err := pipe.Wide(w.columns, w.spirv); err != nil {
			t.Fatal(err)
		}
	}
	set, err := pipe.NewSet([]*Buffer{weights, x, y})
	if err != nil {
		t.Fatal(err)
	}

	for _, width := range []int{1, 2, 4, 8, 16} {
		t.Run(itoa(width), func(t *testing.T) {
			for i := range y.Floats() {
				y.Floats()[i] = 0
			}
			push := matvecKPush{Dim: uint32(m.Rows), FFN: uint32(m.Cols)}
			prog, err := d.Compile(func(r *Recorder) {
				groups := matvecGroups(m.Rows)
				if width == 1 {
					r.Dispatch(set, groups, unsafe.Pointer(&push))
					return
				}
				r.DispatchWide(set, width, groups, unsafe.Pointer(&push))
			})
			if err != nil {
				t.Fatal(err)
			}
			defer prog.Close()
			if err := prog.Run(); err != nil {
				t.Fatal(err)
			}

			// The two sum the same products in different orders, and the card
			// folds the minimum into each block where nn carries a second
			// accumulator, so the demand is the reference tests' and not a bit.
			got := y.Floats()
			var worst, scale float64
			var atCol, atRow int
			for c := 0; c < width; c++ {
				for i := 0; i < m.Rows; i++ {
					scale = math.Max(scale, math.Abs(float64(want[c][i])))
					if gap := math.Abs(float64(got[c*m.Rows+i] - want[c][i])); gap > worst {
						worst, atCol, atRow = gap, c, i
					}
				}
			}
			if worst > 1e-3*scale {
				t.Fatalf("column %d row %d: %v against the processor's %v, %g of a peak of %g",
					atCol, atRow, got[atCol*m.Rows+atRow], want[atCol][atRow], worst, scale)
			}
			t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, width, worst, scale)
		})
	}
}

// kQuantAgainstQ8_0 is the product the tiled kernel actually computes: the weights
// dequantized exactly, against the activation in the Q8_0 form the card holds
// it in.
//
// nn.Matrix.MatVecBatch is not that product. Its Q4_K kernel dequantizes a row
// and dots it against the batch's *floats*, where the Q4_0 one dots against the
// batch's Q8_0 — so for this format alone the processor is the more precise of
// the two and the gap between them is the activation's own rounding, not the
// kernel's. Measured on the 4B's ffn_gate it is 0.0251 of a peak of 4.38, which
// is a part in a hundred and seventy: comparing against the float reference
// would need a tolerance loose enough to pass a genuinely wrong kernel. Against
// this one the demand is a part in a thousand, and what is left is the fp16 the
// cooperative tile stages both operands in.
func kQuantAgainstQ8_0(m nn.Matrix, batch *nn.Batch, columns int) [][]float32 {
	blocks := m.Cols / nn.QuantBlock
	x := make([][]float32, columns)
	for c := 0; c < columns; c++ {
		x[c] = make([]float32, m.Cols)
		for b := 0; b < blocks; b++ {
			scale := batch.Scales[b*batch.Stride+c]
			for i := 0; i < nn.QuantBlock; i++ {
				x[c][b*nn.QuantBlock+i] = float32(batch.Q[(b*batch.Stride+c)*nn.QuantBlock+i]) * scale
			}
		}
	}
	want := make([][]float32, columns)
	for c := range want {
		want[c] = make([]float32, m.Rows)
	}
	row := make([]float32, m.Cols)
	stride, dequant := m.Cols/nn.SuperBlock*144, nn.DequantizeQ4_K
	switch m.Quant {
	case nn.Q6_K:
		stride, dequant = m.Cols/nn.SuperBlock*210, nn.DequantizeQ6_K
	case nn.Q3_K:
		stride, dequant = m.Cols/nn.SuperBlock*110, nn.DequantizeQ3_K
	case nn.Q2_K:
		stride, dequant = m.Cols/nn.SuperBlock*84, nn.DequantizeQ2_K
	}
	for i := 0; i < m.Rows; i++ {
		dequant(m.Data[i*stride:(i+1)*stride], m.Cols, row)
		for c := 0; c < columns; c++ {
			var sum float32
			for j := range row {
				sum += row[j] * x[c][j]
			}
			want[c][i] = sum
		}
	}
	return want
}

// kQuantAgainstFloats is the same product against the batch's float columns,
// which is what a mat-vec here reads.
func kQuantAgainstFloats(m nn.Matrix, batch *nn.Batch, columns int) [][]float32 {
	want := make([][]float32, columns)
	for c := range want {
		want[c] = make([]float32, m.Rows)
	}
	row := make([]float32, m.Cols)
	stride, dequant := m.Cols/nn.SuperBlock*144, nn.DequantizeQ4_K
	switch m.Quant {
	case nn.Q6_K:
		stride, dequant = m.Cols/nn.SuperBlock*210, nn.DequantizeQ6_K
	case nn.Q3_K:
		stride, dequant = m.Cols/nn.SuperBlock*110, nn.DequantizeQ3_K
	case nn.Q2_K:
		stride, dequant = m.Cols/nn.SuperBlock*84, nn.DequantizeQ2_K
	}
	for i := 0; i < m.Rows; i++ {
		dequant(m.Data[i*stride:(i+1)*stride], m.Cols, row)
		for c := 0; c < columns; c++ {
			var sum float32
			for j := range row {
				sum += row[j] * batch.F[c][j]
			}
			want[c][i] = sum
		}
	}
	return want
}

// TestVulkanQ4KTiledMatchesCPU is the same weights through the cooperative
// tiled product, which is what a prompt pass reaches and a token never does.
//
// It stages a weight as fp16 rather than dotting it as an integer, so it is a
// different reading of the same bytes and not the same code at another width.
func TestVulkanQ4KTiledMatchesCPU(t *testing.T) {
	g, m := aQ4_K(t)
	defer g.Close()
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("no cooperative matrices on this device")
	}

	for _, columns := range []int{64, 128, 256, 512} {
		t.Run(itoa(columns), func(t *testing.T) {
			batch := columnsOf(m.Cols, columns)
			want := kQuantAgainstQ8_0(m, batch, columns)

			mm, err := NewMatMulQuant(d, m.Data, m.Rows, m.Cols, columns, true, nn.Q4_K)
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
			if worst > 1e-3*scale {
				t.Fatalf("column %d row %d: %v against the processor's %v, %g of a peak of %g",
					atCol, atRow, got[atCol][atRow], want[atCol][atRow], worst, scale)
			}
			t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, columns, worst, scale)
		})
	}
}

// TestVulkanQ6KMatVecMatchesCPU is shaders/matvec_q6k.comp at every width.
//
// Q6_K is the format whose file order is genuinely awkward — four consecutive
// weights come from two bytes sixty-four apart and two bit positions of a third
// — so this is where a packing that walked it wrongly shows, and it shows on
// every row rather than on a few.
func TestVulkanQ6KMatVecMatchesCPU(t *testing.T) {
	g, m := aQ6_K(t)
	defer g.Close()
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()

	const columns = 16
	batch := columnsOf(m.Cols, columns)
	// Against the floats, which is what this kernel reads. nn's Q6_K kernel
	// dots against the batch's Q8_K form — the tier is the one format here
	// whose processor path is integer — so comparing to MatVecBatch would be
	// comparing two different products and the gap would be that activation's
	// rounding, at a part in two hundred.
	want := kQuantAgainstFloats(m, batch, columns)

	weights, err := d.Upload(splitQ6_K(m.Data, m.Rows, m.Cols))
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	x, err := d.Host(m.Cols*columns*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	xs := x.Floats()
	for c := 0; c < columns; c++ {
		copy(xs[c*m.Cols:], batch.F[c])
	}
	y, err := d.Readback(m.Rows*columns*4, bufferUsageStorage)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Close()

	pipe, err := d.NewPipeline(matvecQ6KSPIRV, 3, uint32(unsafe.Sizeof(matvecKPush{})))
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	for _, w := range []struct {
		columns int
		spirv   []byte
	}{
		{2, matvecQ6K_2SPIRV}, {4, matvecQ6K_4SPIRV}, {8, matvecQ6K_8SPIRV}, {16, matvecQ6K_16SPIRV},
	} {
		if err := pipe.Wide(w.columns, w.spirv); err != nil {
			t.Fatal(err)
		}
	}
	set, err := pipe.NewSet([]*Buffer{weights, x, y})
	if err != nil {
		t.Fatal(err)
	}

	for _, width := range []int{1, 2, 4, 8, 16} {
		t.Run(itoa(width), func(t *testing.T) {
			for i := range y.Floats() {
				y.Floats()[i] = 0
			}
			push := matvecKPush{Dim: uint32(m.Rows), FFN: uint32(m.Cols)}
			prog, err := d.Compile(func(r *Recorder) {
				groups := matvecGroups(m.Rows)
				if width == 1 {
					r.Dispatch(set, groups, unsafe.Pointer(&push))
					return
				}
				r.DispatchWide(set, width, groups, unsafe.Pointer(&push))
			})
			if err != nil {
				t.Fatal(err)
			}
			defer prog.Close()
			if err := prog.Run(); err != nil {
				t.Fatal(err)
			}

			got := y.Floats()
			var worst, scale float64
			var atCol, atRow int
			for c := 0; c < width; c++ {
				for i := 0; i < m.Rows; i++ {
					scale = math.Max(scale, math.Abs(float64(want[c][i])))
					if gap := math.Abs(float64(got[c*m.Rows+i] - want[c][i])); gap > worst {
						worst, atCol, atRow = gap, c, i
					}
				}
			}
			if worst > 1e-3*scale {
				t.Fatalf("column %d row %d: %v against the processor's %v, %g of a peak of %g",
					atCol, atRow, got[atCol*m.Rows+atRow], want[atCol][atRow], worst, scale)
			}
			t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, width, worst, scale)
		})
	}
}

// TestVulkanQ6KTiledMatchesCPU is the six-bit weights through the cooperative
// product, against the same Q8_0 activation reference the Q4_K one uses.
func TestVulkanQ6KTiledMatchesCPU(t *testing.T) {
	g, m := aQ6_K(t)
	defer g.Close()
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	defer d.Close()
	if !d.Coopmat() {
		t.Skip("no cooperative matrices on this device")
	}

	for _, columns := range []int{64, 128, 256, 512} {
		t.Run(itoa(columns), func(t *testing.T) {
			batch := columnsOf(m.Cols, columns)
			want := kQuantAgainstQ8_0(m, batch, columns)

			mm, err := NewMatMulQuant(d, m.Data, m.Rows, m.Cols, columns, true, nn.Q6_K)
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
			if worst > 1e-3*scale {
				t.Fatalf("column %d row %d: %v against the processor's %v, %g of a peak of %g",
					atCol, atRow, got[atCol][atRow], want[atCol][atRow], worst, scale)
			}
			t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, columns, worst, scale)
		})
	}
}
