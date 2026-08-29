package compress

import (
	"math"
	"os"
	"testing"

	"github.com/ThiraSoft/golem/tensors"
)

// What a learned decoder, or a low-rank term, or any transform that hopes to
// find structure in a weight matrix would have to work with: the singular value
// spectrum. A rotation is orthogonal and leaves it alone, so this is what the
// weights are before and after.
//
// The question it answers is whether a rank-r term is worth its bits. At 3.25
// bits a weight, a 1024x3072 matrix costs 1.25 MiB; a rank-64 term in fp16
// costs 0.5 MiB, forty percent more. It has to carry forty percent of the
// energy to break even, and a matrix whose spectrum is flat carries r/min(m,n)
// — six percent here.
//
// GOLEM_MODEL_QWEN points at the checkpoint; the test says nothing without it.
func TestWeightSpectrumIsFlat(t *testing.T) {
	path := os.Getenv("GOLEM_MODEL_QWEN")
	if path == "" {
		t.Skip("GOLEM_MODEL_QWEN unset")
	}
	g, err := tensors.OpenGGUF(path)
	if err != nil {
		t.Skip(err)
	}
	defer g.Close()
	for _, name := range []string{"blk.13.ffn_down.weight", "blk.13.attn_q.weight"} {
		tn, ok := g.Tensors[name]
		if !ok {
			t.Skipf("%s is not in %s", name, path)
		}
		w, err := tn.F32()
		if err != nil {
			t.Fatal(err)
		}
		cols := tn.Shape[0]
		rows := tn.Elems() / cols
		// Randomised range finding: project the matrix onto a random subspace
		// of rank r, orthonormalise, and measure how much of the matrix lands
		// there. Two power iterations pull the estimate up to the true top-r
		// subspace. This is Halko, Martinsson and Tropp, and it answers the
		// question in five products where an eigendecomposition would take an
		// hour.
		var total float64
		for _, v := range w {
			total += float64(v) * float64(v)
		}
		msg := ""
		for _, r := range []int{16, 64, 256} {
			e := topEnergy(w, rows, cols, r) / total
			msg += "  rank " + itoa(r) + ": " + pct(e) + " of the energy, flat would be " + pct(float64(r)/float64(min(rows, cols))) + ";"
		}
		t.Logf("%s %dx%d%s", name, rows, cols, msg)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// topEnergy is how much of ‖A‖² a rank-r subspace holds.
func topEnergy(w []float32, rows, cols, r int) float64 {
	rng := newRand(9)
	// Y = Aᵀ Ω, cols x r.
	y := make([]float64, cols*r)
	om := make([]float64, rows*r)
	for i := range om {
		om[i] = rng()
	}
	mulT := func(dst, src []float64) { // dst[cols,r] = Aᵀ src[rows,r]
		for i := range dst {
			dst[i] = 0
		}
		for i := 0; i < rows; i++ {
			a := w[i*cols : (i+1)*cols]
			for k := 0; k < r; k++ {
				v := src[i*r+k]
				if v == 0 {
					continue
				}
				for j := range a {
					dst[j*r+k] += float64(a[j]) * v
				}
			}
		}
	}
	mul := func(dst, src []float64) { // dst[rows,r] = A src[cols,r]
		Parallel(rows, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				a := w[i*cols : (i+1)*cols]
				for k := 0; k < r; k++ {
					var s float64
					for j := range a {
						s += float64(a[j]) * src[j*r+k]
					}
					dst[i*r+k] = s
				}
			}
		})
	}
	mulT(y, om)
	for it := 0; it < 2; it++ {
		mul(om, y)
		mulT(y, om)
	}
	qr(y, cols, r)
	// ‖Aᵀ Q‖² is the energy the subspace holds, since Q spans it.
	z := make([]float64, rows*r)
	mul(z, y)
	var e float64
	for _, v := range z {
		e += v * v
	}
	return e
}

// qr orthonormalises the columns of an n x r matrix in place, by modified
// Gram-Schmidt.
func qr(a []float64, n, r int) {
	for k := 0; k < r; k++ {
		for j := 0; j < k; j++ {
			var d float64
			for i := 0; i < n; i++ {
				d += a[i*r+k] * a[i*r+j]
			}
			for i := 0; i < n; i++ {
				a[i*r+k] -= d * a[i*r+j]
			}
		}
		var nn float64
		for i := 0; i < n; i++ {
			nn += a[i*r+k] * a[i*r+k]
		}
		nn = math.Sqrt(nn)
		if nn < 1e-12 {
			continue
		}
		for i := 0; i < n; i++ {
			a[i*r+k] /= nn
		}
	}
}

// newRand is a small normal generator, so the test carries no seed but its own.
func newRand(seed uint64) func() float64 {
	s := seed
	return func() float64 {
		var sum float64
		for i := 0; i < 6; i++ {
			s = s*6364136223846793005 + 1442695040888963407
			sum += float64(s>>11) / float64(1<<53)
		}
		return sum - 3
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func pct(x float64) string {
	v := int(math.Round(x * 1000))
	return itoa(v/10) + "." + itoa(v%10) + "%"
}
