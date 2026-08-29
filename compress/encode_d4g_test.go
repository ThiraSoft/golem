package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// The encoder and the kernel have to agree about what a file means. This holds
// one to the other on the only thing that matters: the product.
//
// It is the test that catches a scheme that is self-consistent and wrong — the
// rotation applied to the weights but not the activations, a reciprocal taken
// once too often, a scale folded into the wrong side. Each of those leaves the
// weights looking plausible and the answer somebody else's.
func TestD4GProductSurvivesTheRoundTrip(t *testing.T) {
	const rows, cols = 96, 256
	r := rand.New(rand.NewSource(5))

	// Weights with the shape the rotation exists for: a handful of columns
	// carrying far more than the rest. Gaussian noise would not do — it is
	// already incoherent, so a rotation can only fail to help, and the
	// assertion below would be measuring nothing.
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64()) * 0.02
	}
	for j := 0; j < cols; j += 23 {
		for i := 0; i < rows; i++ {
			w[i*cols+j] *= 8
		}
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}

	// The per-column vector, signs and a salience scale together.
	q := make([]float32, cols)
	pre := make([]float32, cols)
	for j := range q {
		s := float32(0.5 + r.Float64())
		if r.Intn(2) == 0 {
			s = -s
		}
		q[j] = s
		pre[j] = 1 / s
	}

	var unrotated float64
	for _, group := range []int{0, 128} {
		params := D4Params{Beta: 2, ScaleBlock: 64, HadGroup: group, SearchScale: true}
		data := EncodeD4G(w, rows, cols, q, params, nil)
		if want := rows * cols / nn.D4Block * 26; len(data) != want {
			t.Fatalf("group %d: %d bytes, want %d", group, len(data), want)
		}

		// What the kernel does: prepare the activation, then the product.
		xp := make([]float32, cols)
		copy(xp, x)
		nn.PrepareD4G(xp, pre, group)

		m := nn.Matrix{Data: data, Quant: nn.D4G, Rows: rows, Cols: cols}
		b := nn.NewBatch(cols, 1)
		copy(b.F[0], xp)
		got := make([]float32, rows)
		m.MatVec(b, got)

		// What it should have been.
		var num, den float64
		for i := 0; i < rows; i++ {
			var want float64
			for j := 0; j < cols; j++ {
				want += float64(w[i*cols+j]) * float64(x[j])
			}
			d := want - float64(got[i])
			num += d * d
			den += want * want
		}
		rel := math.Sqrt(num / den)
		// This is a wiring check, not a quality bar. The per-column vector here
		// is random rather than a real salience, which is the worst case for it
		// — the error on a column the weights were shrunk into comes back
		// multiplied. What it has to catch is a scheme that is self-consistent
		// and wrong, and those miss by a factor, not by a few percent.
		if rel > 0.35 {
			t.Errorf("group %d: the product is %.4f away from the real one", group, rel)
		}
		if group == 0 {
			unrotated = rel
		} else if rel > unrotated {
			t.Errorf("the rotation made it worse: %.4f rotated against %.4f plain", rel, unrotated)
		}
		t.Logf("group %d: relative error %.4f", group, rel)
	}
}

// A matrix with no vector and no rotation is the plain case the embedding
// table takes, and it has to work on its own.
func TestD4GWithoutRotationOrScaling(t *testing.T) {
	const rows, cols = 32, 128
	r := rand.New(rand.NewSource(9))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64()) * 0.05
	}
	data := EncodeD4G(w, rows, cols, nil, D4Params{Beta: 2, ScaleBlock: 64, SearchScale: true}, nil)

	out := make([]float32, cols)
	m := nn.Matrix{Data: data, Quant: nn.D4G, Rows: rows, Cols: cols}
	var num, den float64
	for i := 0; i < rows; i++ {
		m.Row(i, out)
		for j := 0; j < cols; j++ {
			d := float64(w[i*cols+j] - out[j])
			num += d * d
			den += float64(w[i*cols+j]) * float64(w[i*cols+j])
		}
	}
	if rel := math.Sqrt(num / den); rel > 0.18 {
		t.Errorf("a row read back is %.4f away from the one written", rel)
	}
}

// The wide tier has to cost what it says and buy what it costs. Sixteen bits a
// code against twelve is 34 bytes a block against 26, a whole bit a weight, and
// a bit a weight is six decibels — so anything much under a factor of two on
// the error means the shell of 65536 points is not being reached into.
func TestWideTierCostsAndBuys(t *testing.T) {
	const rows, cols = 64, 512
	r := rand.New(rand.NewSource(13))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64()) * 0.02
	}
	base := D4Params{Beta: 2, ScaleBlock: 32, SearchScale: true}
	var errs [2]float64
	for i, bits := range []int{nn.D4Bits, nn.D4Bits16} {
		p := base
		p.Bits = bits
		data := EncodeD4G(w, rows, cols, nil, p, nil)
		want := rows * cols / nn.D4Block * nn.D4BlockBytes(bits)
		if len(data) != want {
			t.Fatalf("%d bits: %d bytes, want %d", bits, len(data), want)
		}
		errs[i] = RelErrD4G(w, rows, cols, nil, p, data)
	}
	gain := 20 * math.Log10(errs[0]/errs[1])
	t.Logf("relative error %.4f at twelve bits, %.4f at sixteen — %.2f dB for one bit a weight",
		errs[0], errs[1], gain)
	if gain < 4 {
		t.Errorf("the wide tier bought %.2f dB for a whole bit, which is not enough to be worth it", gain)
	}
}
