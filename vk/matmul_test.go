package vk

// The tiled product against the CPU kernel it is meant to replace for a batch.
//
// On a real tensor rather than a synthetic one, for the reason vk/q6k_test.go
// gives: the weights carry every scale pattern the quantizer produces, and a
// shader that unpacked one nibble the wrong way round would still agree with a
// random matrix on most rows.

import (
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// aQ4_0 opens one Q4_0 matrix out of whichever checkpoint this machine has.
func aQ4_0(tb testing.TB) (*tensors.GGUF, nn.Matrix) {
	return namedQ4_0(tb, "blk.0.ffn_down.weight")
}

func namedQ4_0(tb testing.TB, tensor string) (*tensors.GGUF, nn.Matrix) {
	tb.Helper()
	for _, spec := range []struct{ env, tensor string }{
		{"GOLEM_MODEL_QWEN_Q4", tensor},
		{"GOLEM_MODEL_12B", tensor},
		{"GOLEM_MODEL_26B", tensor},
	} {
		path := os.Getenv(spec.env)
		if path == "" {
			continue
		}
		g, err := tensors.OpenGGUF(path)
		if err != nil {
			continue
		}
		t, err := g.Get(spec.tensor)
		if err != nil || t.DType != "Q4_0" {
			g.Close()
			continue
		}
		return g, nn.Matrix{Data: t.Raw, Quant: nn.Q4_0, Cols: t.Shape[0], Rows: t.Shape[1]}
	}
	tb.Skip("set GOLEM_MODEL_QWEN_Q4, GOLEM_MODEL_12B or GOLEM_MODEL_26B to run the tiled product tests")
	return nil, nn.Matrix{}
}

// columns is a batch of repeatable hidden states, quantized the way the engine
// does. Each column is different: a kernel that read one column for all of
// them would pass a test where they were alike.
func columnsOf(width, n int) *nn.Batch {
	batch := nn.NewBatch(width, n)
	for c := 0; c < n; c++ {
		for i := range batch.F[c] {
			batch.F[c][i] = float32(math.Sin(float64(i)*0.37+float64(c)*1.7)) * float32(1+(i+c)%17) * 0.11
		}
		batch.QuantizeColumnRange(c, 0, width)
	}
	return batch
}

// oneColumn is that batch's column c on its own, in the layout a device buffer
// wants: nn.Batch interleaves its quantized form by block and then by column,
// so a column of it is contiguous only when there is one.
func oneColumn(b *nn.Batch, c int) *nn.Batch {
	one := nn.NewBatch(b.Width, 1)
	copy(one.F[0], b.F[c])
	one.QuantizeColumnRange(0, 0, b.Width)
	return one
}

func TestMatMulMatchesCPU(t *testing.T) {
	for _, coop := range []bool{false, true} {
		name := "tiled"
		if coop {
			name = "coopmat"
		}
		t.Run(name, func(t *testing.T) { matMulMatchesCPU(t, coop) })
	}
}

func matMulMatchesCPU(t *testing.T, coop bool) {
	g, m := aQ4_0(t)
	defer g.Close()
	d := open(t)
	defer d.Close()

	const cols = 32
	batch := columnsOf(m.Cols, cols)

	// What the CPU makes of it, one product per column.
	want := make([][]float32, cols)
	for c := range want {
		want[c] = make([]float32, m.Rows)
	}
	m.MatVecBatch(batch, want)

	mm, err := NewMatMul(d, m.Data, m.Rows, m.Cols, cols, coop)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	for c := 0; c < cols; c++ {
		if err := mm.SetColumn(c, oneColumn(batch, c)); err != nil {
			t.Fatal(err)
		}
	}
	got := make([][]float32, cols)
	for c := range got {
		got[c] = make([]float32, m.Rows)
	}
	if err := mm.Run(got); err != nil {
		t.Fatal(err)
	}

	// The two sum the same products in different orders — this folds the
	// recentring into each block the way ggml does, where nn carries a second
	// accumulator — so the tolerance is the reference tests' rather than a
	// bit-for-bit one.
	var worst float64
	var at, where int
	var scale float64
	for c := range got {
		for i := range got[c] {
			scale = math.Max(scale, math.Abs(float64(want[c][i])))
			if gap := math.Abs(float64(got[c][i] - want[c][i])); gap > worst {
				worst, at, where = gap, c, i
			}
		}
	}
	if worst > 1e-3*scale {
		t.Fatalf("column %d row %d: %v against the CPU's %v, %g of a peak of %g",
			at, where, got[at][where], want[at][where], worst, scale)
	}
	t.Logf("%d rows by %d columns, worst gap %g of a peak of %g", m.Rows, cols, worst, scale)
}

// BenchmarkMatMul is the tiled product on one matrix, a pass at a time, with
// nothing else in the submission. What it says is the bandwidth the kernel
// reaches: the matrix is read once whatever the width of the pass, so a pass
// that costs the same at thirty-two columns as at one is a pass that has done
// thirty-two times the work for nothing.
func BenchmarkMatMul(b *testing.B) {
	for _, shape := range []struct {
		name   string
		tensor string
	}{
		{"down", "blk.0.ffn_down.weight"}, // 2560 rows by 9728: forty workgroups
		{"gate", "blk.0.ffn_gate.weight"}, // 9728 rows by 2560: a hundred and fifty-two
	} {
		for _, spec := range []struct {
			name    string
			columns int
			coop    bool
		}{{"8", 8, false}, {"32", 32, false}, {"coop32", 32, true}, {"coop64", 64, true}, {"coop128", 128, true}, {"coop256", 256, true}} {
			columns, coop := spec.columns, spec.coop
			b.Run(shape.name+"/"+spec.name, func(b *testing.B) {
				g, m := namedQ4_0(b, shape.tensor)
				defer g.Close()
				d := open(b)
				defer d.Close()

				mm, err := NewMatMul(d, m.Data, m.Rows, m.Cols, columns, coop)
				if err != nil {
					b.Fatal(err)
				}
				defer mm.Close()

				batch := columnsOf(m.Cols, columns)
				for c := 0; c < columns; c++ {
					if err := mm.SetColumn(c, oneColumn(batch, c)); err != nil {
						b.Fatal(err)
					}
				}
				out := make([][]float32, columns)
				for c := range out {
					out[c] = make([]float32, m.Rows)
				}
				if err := mm.Run(out); err != nil {
					b.Fatal(err)
				}
				// Sixty-four passes to a submission, so that the card is not
				// allowed to rest between them.
				const times = 64
				bytes := m.Rows * rowBytesQ4_0(m.Cols)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := mm.RunTimes(times); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				seconds := b.Elapsed().Seconds() / float64(b.N) / times
				b.ReportMetric(float64(bytes)/seconds/1e9, "GB/s")
				b.ReportMetric(seconds*1e6/float64(columns), "us/column")
			})
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for n > 0 {
		d = append([]byte{byte('0' + n%10)}, d...)
		n /= 10
	}
	return string(d)
}
