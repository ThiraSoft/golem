package vk

// The trellis kernel against the processor's, on bytes a real encoder wrote.
//
// The exactness is the point, and it is a stronger contract than the encoders
// keep with each other. Two Viterbis that find different minimum-cost paths
// write different files and both are right; two decoders that disagree about
// what a file says are two formats sharing a name. compress/README.md's
// standing lesson is that such a disagreement changes no shape and no name —
// the model loads, every tensor is the size it should be, and it answers
// nonsense.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/compress"
	"github.com/ThiraSoft/golem/nn"
)

func t4gMatrix(tb testing.TB, rows, cols int) ([]byte, []float32) {
	return t4gMatrixAs(tb, rows, cols, nn.T4G)
}

func t4gMatrixAs(tb testing.TB, rows, cols int, kind nn.Quant) ([]byte, []float32) {
	tb.Helper()
	r := rand.New(rand.NewSource(23))
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
	data := compress.EncodeT4GAs(w, rows, cols, q, compress.D4Params{
		ScaleBlock: nn.T4GBlock, HadGroup: 128}, kind)
	return data, q
}

// Every weight of every position of a sequence, exactly.
//
// A one-hot activation turns the product into a single weight: every other term
// is a multiplication by zero and adding zero to a float is exact, so what the
// kernel writes is the weight it decoded and nothing else. Sweeping the hot
// column over a whole row walks all 128 offsets of a path, both halves of both
// step codes, and both alignments a twelve-bit window can have inside a byte
// pair — which is the whole of what there is to get wrong.
func TestT4GDecodeMatchesCPUExactly(t *testing.T) {
	testGolemDecodeMatchesCPUExactly(t, nn.T4G)
	testGolemDecodeMatchesCPUExactly(t, nn.T5G)
}

// testGolemDecodeMatchesCPUExactly is the sweep every trellis tier is held to,
// factored so a new tier gets it by calling in rather than by copying it —
// two copies of an exactness sweep drift apart.
func testGolemDecodeMatchesCPUExactly(t *testing.T, kind nn.Quant) {
	const rows, cols = 32, 256
	d := open(t)
	defer d.Close()

	data, _ := t4gMatrixAs(t, rows, cols, kind)
	m := nn.Matrix{Data: data, Quant: kind, Rows: rows, Cols: cols}
	want := make([]float32, cols)

	gpu, err := newHostGolemMatrix(d, data, rows, cols, kind)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()

	x := make([]float32, cols)
	got := make([]float32, rows)
	for j := 0; j < cols; j++ {
		for i := range x {
			x[i] = 0
		}
		x[j] = 1
		if err := gpu.MatVec(x, got); err != nil {
			t.Fatal(err)
		}
		for r := 0; r < rows; r++ {
			m.Row(r, want)
			if got[r] != want[j] {
				t.Fatalf("%s weight [%d,%d]: the card reads %v, the processor %v",
					kind, r, j, got[r], want[j])
			}
		}
	}
}

func TestT4GMatVecMatchesCPU(t *testing.T) {
	const rows, cols = 512, 1024
	d := open(t)
	defer d.Close()

	data, q := t4gMatrix(t, rows, cols)
	m := nn.Matrix{Data: data, Quant: nn.T4G, Rows: rows, Cols: cols}

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

	gpu, err := newHostGolemMatrix(d, data, rows, cols, nn.T4G)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	got := make([]float32, rows)
	if err := gpu.MatVec(x, got); err != nil {
		t.Fatal(err)
	}

	var scale, worst float64
	var where int
	for _, v := range want {
		scale = math.Max(scale, math.Abs(float64(v)))
	}
	for i := range want {
		if e := math.Abs(float64(got[i] - want[i])); e/scale > worst {
			worst, where = e/scale, i
		}
	}
	// The two sides decode identically — the test above says so weight by
	// weight — so what is left is the order of a thousand float additions.
	if worst > 1e-5 {
		t.Fatalf("row %d diverges: CPU %v, GPU %v (relative %g)", where, want[where], got[where], worst)
	}
	t.Logf("%d rows of %d, worst gap %g of the largest output", rows, cols, worst)
}

// A row of the format is 67·cols/128 bytes, which is odd whenever the row is
// not a multiple of 512 wide. A vision tower's is 1152, so this is not
// hypothetical: the upload has to round up to a word or the last bytes of the
// last row are read out of a word past the end of the buffer.
func TestT4GUnalignedRowsDecode(t *testing.T) {
	const rows, cols = 8, 1152
	d := open(t)
	defer d.Close()

	data, _ := t4gMatrix(t, rows, cols)
	if nn.T4GRowBytes(cols)%4 == 0 {
		t.Fatalf("%d columns give a word-aligned row; this test is not testing anything", cols)
	}
	m := nn.Matrix{Data: data, Quant: nn.T4G, Rows: rows, Cols: cols}
	gpu, err := newHostGolemMatrix(d, data, rows, cols, nn.T4G)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()

	want := make([]float32, cols)
	got := make([]float32, rows)
	x := make([]float32, cols)
	// The last weight of the last row is the one an unrounded upload loses.
	for _, j := range []int{0, 1, cols / 2, cols - 2, cols - 1} {
		for i := range x {
			x[i] = 0
		}
		x[j] = 1
		if err := gpu.MatVec(x, got); err != nil {
			t.Fatal(err)
		}
		for r := 0; r < rows; r++ {
			m.Row(r, want)
			if got[r] != want[j] {
				t.Fatalf("weight [%d,%d]: the card reads %v, the processor %v", r, j, got[r], want[j])
			}
		}
	}
}
