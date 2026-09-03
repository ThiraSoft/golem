package vk

// The general head against the product it is supposed to compute, on real
// tensors.
//
// The reference is not the CPU kernel. Those kernels do not all compute the
// same product: nn's Q8_0 mat-vec reads the activation as float32 where every
// kernel behind the format door reads its eight-bit form, and nn's Q6_K one
// reads the Q8_K form where the door's Q6_K kernel reads Q8_0 — so comparing
// against them measures the activation's quantization and calls it an error.
// Two per cent of one, on the checkpoints here.
//
// What the door's kernels do compute is exact in float64 from the same bytes:
// the dequantized row against the dequantized activation, where the activation
// is the eight-bit one the kernel was handed and not the floats it came from.
// That reference is the same for every format, which is what makes this one
// test rather than eight, and it is tight — a shader that unpacks a nibble
// from the wrong half is out by whole units against it.
//
// It runs over whichever checkpoints this machine has, and each is a different
// format, so the table is coverage of the door rather than of any one model.

import (
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// headOf opens a checkpoint's head, which is its output matrix where it has
// one and its tied input embedding where it does not.
func headOf(tb testing.TB, path string) (*tensors.GGUF, nn.Matrix) {
	tb.Helper()
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		tb.Fatal(err)
	}
	t, err := g.Get("output.weight")
	if err != nil {
		if t, err = g.Get("token_embd.weight"); err != nil {
			g.Close()
			tb.Fatalf("%s has neither an output matrix nor a tied embedding", path)
		}
	}
	q, ok := nn.QuantOf(t.DType)
	if !ok {
		g.Close()
		tb.Skipf("%s: the head is %s, which nn does not read", path, t.DType)
	}
	return g, nn.Matrix{Data: t.Raw, Quant: q, Cols: t.Shape[0], Rows: t.Shape[1]}
}

// headCheckpoints is every checkpoint this machine offers, by the variable
// that names it. A variable that is not set skips its row.
var headCheckpoints = []string{
	"GOLEM_MODEL", "GOLEM_MODEL_12B", "GOLEM_MODEL_26B", "GOLEM_MODEL_26B_Q8",
	"GOLEM_MODEL_QWEN_Q4", "GOLEM_MODEL_Q4KM", "GOLEM_MODEL_Q3KM",
}

func TestQuantHeadComputesItsProduct(t *testing.T) {
	ran := 0
	for _, env := range headCheckpoints {
		path := os.Getenv(env)
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err != nil {
			continue
		}
		ran++
		t.Run(env, func(t *testing.T) {
			g, m := headOf(t, path)
			defer g.Close()
			if !QuantReadable(m.Quant) {
				t.Skipf("%s heads have no kernel here", m.Quant)
			}
			d := open(t)
			defer d.Close()

			h, err := NewQuantHead(d, m.Quant, m.Data, m.Rows, m.Cols)
			if err != nil {
				t.Fatal(err)
			}
			defer h.Close()

			b := nn.NewBatch(m.Cols, 1)
			for i := range b.F[0] {
				// Something with the shape of a real hidden state: a few large
				// values among many small ones, both signs, no round numbers.
				b.F[0][i] = float32(math.Sin(float64(i)*0.37)) * float32(1+i%17) * 0.11
			}
			h.Prepare(b)

			got := make([]float32, m.Rows)
			if err := h.MatVec(b, 0, got); err != nil {
				t.Fatal(err)
			}

			// The activation the kernel was actually handed, put back as the
			// reals it stands for. This is where the CPU comparison went
			// wrong: these are not b.F.
			act := make([]float64, m.Cols)
			for i := range act {
				act[i] = float64(b.Q[i]) * float64(b.Scales[i/nn.QuantBlock])
			}

			// Sampled rather than exhaustive: a quarter of a million rows of
			// float64 dequantization is minutes, and an unpacking fault is a
			// property of the format and not of one row. The stride is odd so
			// that the sample does not land on the same lane of every
			// workgroup.
			row := make([]float32, m.Cols)
			var worst float64
			var where int
			const samples = 1024
			stride := max(1, m.Rows/samples|1)
			for i := 0; i < m.Rows; i += stride {
				m.Row(i, row)
				var want float64
				for j, w := range row {
					want += float64(w) * act[j]
				}
				scale := math.Max(1, math.Abs(want))
				if rel := math.Abs(float64(got[i])-want) / scale; rel > worst {
					worst, where = rel, i
				}
			}
			// Float32 accumulation over a few thousand terms, in an order the
			// shader chooses and this loop does not. A part in a hundred
			// thousand is that; anything a shader gets structurally wrong is
			// orders above it.
			if worst > 1e-5 {
				t.Errorf("%s head: row %d is off the exact product by a relative %g", m.Quant, where, worst)
			}
			t.Logf("%s, %d rows of %d, %d sampled: worst relative gap %g at row %d",
				m.Quant, m.Rows, m.Cols, (m.Rows+stride-1)/stride, worst, where)
		})
	}
	if ran == 0 {
		t.Skip("no checkpoint named by " + headCheckpoints[0] + " and friends")
	}
}

// BenchmarkQuantHead is the head alone on an otherwise empty card, which is
// the only way to see the kernel rather than the residency: in a whole model
// the same matrix competes for device memory with every block, and a driver
// that cannot fit it says nothing and reads it over the bus instead.
func BenchmarkQuantHead(b *testing.B) {
	for _, env := range headCheckpoints {
		path := os.Getenv(env)
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err != nil {
			continue
		}
		b.Run(env, func(b *testing.B) {
			g, m := headOf(b, path)
			defer g.Close()
			if !QuantReadable(m.Quant) {
				b.Skipf("%s heads have no kernel here", m.Quant)
			}
			d := open(b)
			defer d.Close()
			h, err := NewQuantHead(d, m.Quant, m.Data, m.Rows, m.Cols)
			if err != nil {
				b.Fatal(err)
			}
			defer h.Close()

			batch := nn.NewBatch(m.Cols, 1)
			for i := range batch.F[0] {
				batch.F[0][i] = float32(math.Sin(float64(i)*0.37)) * 0.1
			}
			h.Prepare(batch)
			out := make([]float32, m.Rows)
			rowBytes, err := quantRowBytes(m.Quant, m.Cols)
			if err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(m.Rows) * int64(rowBytes))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := h.MatVec(batch, 0, out); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
