package vk

// The head against the CPU kernel it replaces, on the real tensor.
//
// A parity test on a matrix this size is worth more than a synthetic one: the
// weights carry every scale pattern the quantizer produces, and a shader that
// unpacks one nibble the wrong way round would still agree with a random
// matrix on most rows.

import (
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/ThiraSoft/golem/internal/heavy"
	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// head opens the 26B's tied embedding, which is the only Q6_K tensor in it.
func head(tb testing.TB) (*tensors.GGUF, nn.Matrix) {
	tb.Helper()
	path := os.Getenv("GOLEM_MODEL_26B")
	if path == "" {
		tb.Skip("set GOLEM_MODEL_26B to run the Vulkan head tests")
	}
	// A variable naming a file that is not there is a machine that does not
	// have the model, not a fault in the kernel. gemma's own openers say the
	// same thing the same way.
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("GOLEM_MODEL_26B names %s, which is not there", path)
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		tb.Fatal(err)
	}
	t, err := g.Get("token_embd.weight")
	if err != nil {
		g.Close()
		tb.Fatal(err)
	}
	if t.DType != "Q6_K" {
		g.Close()
		tb.Fatalf("the head is %s, not Q6_K", t.DType)
	}
	m := nn.Matrix{Data: t.Raw, Quant: nn.Q6_K, Cols: t.Shape[0], Rows: t.Shape[1]}
	return g, m
}

// activation is a repeatable hidden state, quantized the way the engine does.
func activation(width int) *nn.Batch {
	b := nn.NewBatch(width, 1)
	for i := range b.F[0] {
		// Something with the shape of a real hidden state: a few large values
		// among many small ones, both signs, no round numbers.
		b.F[0][i] = float32(math.Sin(float64(i)*0.37)) * float32(1+i%17) * 0.11
	}
	b.QuantizeK()
	return b
}

// open is the one door onto the card in this package's tests, and the guard
// sits in it: every test here submits work to a device, which is heavy in the
// sense internal/heavy means — the card runs flat out, and it gets hot.
func open(tb testing.TB) *Device {
	tb.Helper()
	heavy.Skip(tb, "it puts work on the card")
	d, err := Open()
	if err != nil {
		tb.Skipf("no Vulkan compute device: %v", err)
	}
	return d
}

func TestQ6KHeadMatchesCPU(t *testing.T) {
	g, m := head(t)
	defer g.Close()
	d := open(t)
	defer d.Close()

	h, err := NewQ6KHead(d, m.Data, m.Rows, m.Cols)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	b := activation(m.Cols)
	want := make([]float32, m.Rows)
	m.MatVec(b, want)

	got := make([]float32, m.Rows)
	if err := h.MatVec(b, 0, got); err != nil {
		t.Fatal(err)
	}

	var worst float64
	var where int
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		scale := math.Max(1, math.Abs(float64(want[i])))
		if rel := diff / scale; rel > worst {
			worst, where = rel, i
		}
	}
	// The integer part of the product is exact on both sides; only the final
	// float32 multiply by the two scales can land a bit apart.
	if worst > 1e-6 {
		t.Fatalf("row %d diverges: CPU %v, GPU %v (relative %g)", where, want[where], got[where], worst)
	}
	t.Logf("%d rows, worst relative gap %g", m.Rows, worst)
}

func BenchmarkQ6KHeadGPU(b *testing.B) {
	g, m := head(b)
	defer g.Close()
	d := open(b)
	defer d.Close()

	h, err := NewQ6KHead(d, m.Data, m.Rows, m.Cols)
	if err != nil {
		b.Fatal(err)
	}
	defer h.Close()

	batch := activation(m.Cols)
	out := make([]float32, m.Rows)
	b.SetBytes(int64(m.Rows) * int64(m.Cols) / nn.SuperBlock * q6kPaddedBytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := h.MatVec(batch, 0, out); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQ6KHeadCPU(b *testing.B) {
	g, m := head(b)
	defer g.Close()

	batch := activation(m.Cols)
	out := make([]float32, m.Rows)
	b.SetBytes(int64(len(m.Data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.MatVec(batch, out)
	}
}

// BenchmarkQ6KHeadSaturated is the same product with thirty-two dispatches to
// a submission, which is what the kernel costs when the card is awake. The gap
// between this and BenchmarkQ6KHeadGPU is not the shader: it is the round trip
// and the clocks the round trip fails to raise.
func BenchmarkQ6KHeadSaturated(b *testing.B) {
	g, m := head(b)
	defer g.Close()
	d := open(b)
	defer d.Close()

	h, err := NewQ6KHead(d, m.Data, m.Rows, m.Cols)
	if err != nil {
		b.Fatal(err)
	}
	defer h.Close()

	batch := activation(m.Cols)
	out := make([]float32, m.Rows)
	if err := h.MatVec(batch, 0, out); err != nil {
		b.Fatal(err)
	}

	const rounds = 32
	push := q6kPush{rows: uint32(h.rows), superblocks: uint32(h.superblocks)}
	b.SetBytes(int64(m.Rows) * int64(m.Cols) / nn.SuperBlock * q6kPaddedBytes * rounds)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := h.set.DispatchTimes(h.groups, unsafe.Pointer(&push), rounds); err != nil {
			b.Fatal(err)
		}
	}
}
