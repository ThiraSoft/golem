package compress

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ThiraSoft/golem/nn"
)

// The two codebooks in the same container, on the same weights, at the same
// size. TestLatticeAgainstTheBestScalar says the lattice should be 0.36 dB
// ahead on an ideal Gaussian; this asks the same question of the encoder that
// has to write real rows, with the step search and the rotation around it.
func TestL8GAgainstD4G(t *testing.T) {
	const rows, cols = 128, 1024
	r := rand.New(rand.NewSource(37))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64()) * 0.02
	}
	q := RandomSigns(cols, 47)

	p4 := D4Params{Beta: 2, ScaleBlock: 32, HadGroup: 128, SearchScale: true}
	p8 := D4Params{Beta: 1, ScaleBlock: 32, HadGroup: 128, SearchScale: true}
	d4 := EncodeD4G(w, rows, cols, q, p4, nil)
	l8 := EncodeL8G(w, rows, cols, q, p8)
	if len(d4) != len(l8) {
		t.Fatalf("the two formats are %d and %d bytes, and are meant to be the same", len(d4), len(l8))
	}

	e4 := relErr(w, rows, cols, q, p4, d4, nn.D4G)
	e8 := relErr(w, rows, cols, q, p8, l8, nn.L8G)
	t.Logf("%.4f relative error from the lattice (%.2f dB), %.4f from eight levels (%.2f dB) — the lattice buys %.2f dB",
		e4, -20*math.Log10(e4), e8, -20*math.Log10(e8), 20*math.Log10(e8/e4))
	if gain := 20 * math.Log10(e8/e4); gain > 0.8 {
		t.Errorf("the lattice is %.2f dB ahead, which is more than a codebook of the same rate should give", gain)
	}
}

func relErr(w []float32, rows, cols int, q []float32, p D4Params, data []byte, kind nn.Quant) float64 {
	m := nn.Matrix{Data: data, Quant: kind, Rows: rows, Cols: cols}
	row := make([]float32, cols)
	rec := make([]float32, cols)
	var num, den float64
	for r := 0; r < rows; r++ {
		copy(row, w[r*cols:(r+1)*cols])
		nn.PrepareD4G(row, q, p.HadGroup)
		m.Row(r, rec)
		for j := range row {
			d := float64(row[j] - rec[j])
			num += d * d
			den += float64(row[j]) * float64(row[j])
		}
	}
	return math.Sqrt(num / den)
}

// Why the scalar codebook costs the model three times what its weight error
// says it should.
//
// Eight levels reach 2.15 of a block's step and no further. A lattice point may
// have a coordinate out to six of it, because the shell is a ball in four
// dimensions and its radius is spent on the norm of four weights rather than on
// one. So the tail of the distribution is clipped by one and not the other, and
// a clipped weight is a biased error rather than a white one — it always points
// the same way, and twenty-eight layers of it do not cancel.
//
// This counts how much of the tail each codebook flattens.
func TestScalarClipsTheTail(t *testing.T) {
	const rows, cols = 128, 1024
	r := rand.New(rand.NewSource(41))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64()) * 0.02
	}
	q := RandomSigns(cols, 53)
	p8 := D4Params{Beta: 1, ScaleBlock: 32, HadGroup: 128, SearchScale: true}
	p4 := D4Params{Beta: 2, ScaleBlock: 32, HadGroup: 128, SearchScale: true}

	l8 := EncodeL8G(w, rows, cols, q, p8)
	d4 := EncodeD4G(w, rows, cols, q, p4, nil)

	// How often each lands on the edge of what it can say.
	var atEdge8, atEdge4, total int
	m8 := nn.Matrix{Data: l8, Quant: nn.L8G, Rows: rows, Cols: cols}
	m4 := nn.Matrix{Data: d4, Quant: nn.D4G, Rows: rows, Cols: cols}
	row := make([]float32, cols)
	rec8 := make([]float32, cols)
	rec4 := make([]float32, cols)
	var bias8, bias4 float64
	for r := 0; r < rows; r++ {
		copy(row, w[r*cols:(r+1)*cols])
		nn.PrepareD4G(row, q, 128)
		m8.Row(r, rec8)
		m4.Row(r, rec4)
		for j := range row {
			total++
			// The reconstruction falling short of the weight, in the same
			// direction as the weight, is what clipping looks like from here.
			if math.Abs(float64(rec8[j])) < math.Abs(float64(row[j]))*0.75 {
				atEdge8++
			}
			if math.Abs(float64(rec4[j])) < math.Abs(float64(row[j]))*0.75 {
				atEdge4++
			}
			bias8 += float64(row[j]) * float64(row[j]-rec8[j])
			bias4 += float64(row[j]) * float64(row[j]-rec4[j])
		}
	}
	var energy float64
	for _, v := range w {
		energy += float64(v) * float64(v)
	}
	t.Logf("weights the codebook could not reach: %.2f%% with eight levels, %.2f%% with the lattice",
		100*float64(atEdge8)/float64(total), 100*float64(atEdge4)/float64(total))
	t.Logf("shrinkage, as a share of the energy: %.3f%% with eight levels, %.3f%% with the lattice",
		100*bias8/energy, 100*bias4/energy)
}

// Why a lattice beats its own mean squared error.
//
// A dot product does not care about the size of the errors, it cares about the
// size of their sum: the error of y = Σ wⱼxⱼ is Σ εⱼxⱼ. If the εⱼ are
// independent that sum has the variance the mean squared error predicts. If
// they lean against each other it has less.
//
// And they do, for a lattice. A vector quantizer's error is the vector from a
// point to the cell it fell in, and a Voronoi cell is a bounded shape: the four
// components of one error cannot all be large and of the same sign, because
// that would put the point nearer another lattice point. A scalar quantizer has
// no such constraint — each weight is rounded on its own line and the four
// errors know nothing of each other.
//
// This measures the cross terms. Negative means the errors cancel in a product,
// and that a codebook's mean squared error is telling less than the whole story.
func TestLatticeErrorsLeanAgainstEachOther(t *testing.T) {
	const rows, cols = 256, 1024
	r := rand.New(rand.NewSource(43))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(r.NormFloat64()) * 0.02
	}
	q := RandomSigns(cols, 59)
	for _, c := range []struct {
		what string
		kind nn.Quant
		p    D4Params
		data func(D4Params) []byte
	}{
		{"the lattice", nn.D4G, D4Params{Beta: 2, ScaleBlock: 32, HadGroup: 128, SearchScale: true},
			func(p D4Params) []byte { return EncodeD4G(w, rows, cols, q, p, nil) }},
		{"eight levels", nn.L8G, D4Params{Beta: 1, ScaleBlock: 32, HadGroup: 128, SearchScale: true},
			func(p D4Params) []byte { return EncodeL8G(w, rows, cols, q, p) }},
	} {
		m := nn.Matrix{Data: c.data(c.p), Quant: c.kind, Rows: rows, Cols: cols}
		row := make([]float32, cols)
		rec := make([]float32, cols)
		var square, cross float64
		for i := 0; i < rows; i++ {
			copy(row, w[i*cols:(i+1)*cols])
			nn.PrepareD4G(row, q, 128)
			m.Row(i, rec)
			for j := 0; j+4 <= cols; j += 4 {
				var e [4]float64
				for k := 0; k < 4; k++ {
					e[k] = float64(row[j+k] - rec[j+k])
					square += e[k] * e[k]
				}
				for a := 0; a < 4; a++ {
					for b := a + 1; b < 4; b++ {
						cross += 2 * e[a] * e[b]
					}
				}
			}
		}
		t.Logf("%-12s cross terms are %+.1f%% of the squared error", c.what, 100*cross/square)
	}
}
