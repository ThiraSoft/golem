package compress

// Error compensation: spending the columns that are not yet quantized on the
// error of the ones that are.
//
// Rounding a weight matrix column by column throws away the error of each
// column. But the product Wx is what matters, not W, and the columns of W are
// not orthogonal under the activations: if column j comes out too large, some
// combination of the columns after it can absorb most of what that costs the
// output. The combination is read off the Hessian of the activations,
// H = XᵀX, and the classical result — OBS, and GPTQ after it — is that with
// R the Cholesky factor of H⁻¹, quantizing column j and then subtracting
//
//	(wⱼ - qⱼ)/R[j,j] ⊗ R[j, j+1:]
//
// from the remaining columns is the optimal correction, at the price of one
// triangular factor per matrix and one rank-one update per column.
//
// Two things here are not textbook GPTQ. The quantizer is a lattice, so four
// columns are decided at once and the update is by blocks of four rather than
// rank one. And the factor is not taken over the whole matrix but over a
// window of columns: the cost of the factor is cubic in its width, and the
// weights have already been rotated, which spreads what the Hessian knows and
// leaves most of the correlation local. A window of a few hundred columns buys
// nearly all of what the full matrix would, for a hundredth of the arithmetic.

import (
	"math"
)

// Comp is one site's compensation plan: for each window of columns, the upper
// triangular R with H⁻¹ = RᵀR over that window's activations.
type Comp struct {
	Window int
	Cols   int
	blocks [][]float32 // one row-major m×m per window
	sizes  []int
}

// Acc accumulates a site's Hessian one activation at a time, and only the part
// of it the windows read. That is what makes the calibration text as long as we
// like: a stored activation costs its width per token, a windowed Hessian costs
// its width times the window and nothing per token at all.
type Acc struct {
	cols, window int
	blocks       [][]float32
	sizes        []int
	rows         int
}

func NewAcc(cols, window int) *Acc {
	n := (cols + window - 1) / window
	a := &Acc{cols: cols, window: window, blocks: make([][]float32, n), sizes: make([]int, n)}
	for k := range a.blocks {
		m := window
		if k*window+m > cols {
			m = cols - k*window
		}
		a.sizes[k] = m
		a.blocks[k] = make([]float32, m*m)
	}
	return a
}

// AddRows takes a batch of activations, each already scaled and rotated the way
// the weights were. Only the upper triangle of each window is written; Comp
// mirrors it. The work is split by window rather than by row because the
// windows are what the accumulators are, and two rows of the same window would
// have to take turns.
func (a *Acc) AddRows(x [][]float32) {
	a.rows += len(x)
	Parallel(len(a.blocks), func(lo, hi int) {
		for k := lo; k < hi; k++ {
			m := a.sizes[k]
			h := a.blocks[k]
			for _, row := range x {
				v := row[k*a.window : k*a.window+m]
				for i := 0; i < m; i++ {
					vi := v[i]
					if vi == 0 {
						continue
					}
					hi := h[i*m : i*m+m]
					for j := i; j < m; j++ {
						hi[j] += vi * v[j]
					}
				}
			}
		}
	})
}

// Add is one activation.
func (a *Acc) Add(x []float32) { a.AddRows([][]float32{x}) }

// Comp factors what has been accumulated.
func (a *Acc) Comp(damp float64) *Comp {
	if a.rows == 0 {
		return nil
	}
	c := &Comp{Window: a.window, Cols: a.cols,
		blocks: make([][]float32, len(a.blocks)), sizes: a.sizes}
	Parallel(len(a.blocks), func(lo, hi int) {
		for k := lo; k < hi; k++ {
			m := a.sizes[k]
			h := a.blocks[k]
			for i := 0; i < m; i++ {
				for j := 0; j < i; j++ {
					h[i*m+j] = h[j*m+i]
				}
			}
			c.blocks[k] = factor(h, m, damp)
		}
	})
	return c
}

// NewComp is Acc over a batch of activations, which is what a test has and a
// converter does not.
func NewComp(x [][]float32, cols, window int, damp float64) *Comp {
	if window <= 0 || len(x) == 0 {
		return nil
	}
	a := NewAcc(cols, window)
	a.AddRows(x)
	return a.Comp(damp)
}

// block is the factor for the window a column falls in, and where that window
// starts.
func (c *Comp) block(k int) ([]float32, int) { return c.blocks[k], c.sizes[k] }

// factor turns a Hessian into the R the sweep reads. The path is the one GPTQ
// takes — factor H, invert through that factor, factor the inverse — and the
// ridge climbs until the first of those succeeds, because a Hessian built from
// fewer tokens than it has columns is singular and no amount of care makes it
// otherwise.
func factor(h []float32, m int, damp float64) []float32 {
	var mean float64
	for i := 0; i < m; i++ {
		mean += float64(h[i*m+i])
	}
	mean /= float64(m)
	if mean <= 0 {
		mean = 1
	}
	l := make([]float32, m*m)
	lambda := damp
	for {
		copy(l, h)
		for i := 0; i < m; i++ {
			l[i*m+i] += float32(lambda * mean)
		}
		if cholLower(l, m) {
			break
		}
		lambda *= 4
		if lambda > 1e3 {
			// Nothing here to compensate with: give the sweep the identity,
			// which makes every correction zero.
			r := make([]float32, m*m)
			for i := 0; i < m; i++ {
				r[i*m+i] = 1
			}
			return r
		}
	}
	inv := inverseFromChol(l, m)
	if !cholUpper(inv, m) {
		for i := range inv {
			inv[i] = 0
		}
		for i := 0; i < m; i++ {
			inv[i*m+i] = 1
		}
	}
	return inv
}

// cholLower overwrites the lower triangle of a with L, where a = LLᵀ, and says
// whether a was positive definite.
func cholLower(a []float32, n int) bool {
	for i := 0; i < n; i++ {
		ri := a[i*n : i*n+n]
		for j := 0; j <= i; j++ {
			rj := a[j*n : j*n+n]
			var s float64
			for k := 0; k < j; k++ {
				s += float64(ri[k]) * float64(rj[k])
			}
			v := float64(ri[j]) - s
			if i == j {
				if v <= 0 {
					return false
				}
				ri[j] = float32(math.Sqrt(v))
			} else {
				ri[j] = float32(v / float64(rj[j]))
			}
		}
		for j := i + 1; j < n; j++ {
			ri[j] = 0
		}
	}
	return true
}

// inverseFromChol builds A⁻¹ from the L of A = LLᵀ, by inverting L and forming
// L⁻ᵀL⁻¹.
func inverseFromChol(l []float32, n int) []float32 {
	iv := make([]float32, n*n) // lower triangular L⁻¹
	for j := 0; j < n; j++ {
		iv[j*n+j] = 1 / l[j*n+j]
		for i := j + 1; i < n; i++ {
			var s float64
			for k := j; k < i; k++ {
				s += float64(l[i*n+k]) * float64(iv[k*n+j])
			}
			iv[i*n+j] = float32(-s / float64(l[i*n+i]))
		}
	}
	out := make([]float32, n*n)
	Parallel(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			for j := i; j < n; j++ {
				var s float64
				for k := j; k < n; k++ {
					s += float64(iv[k*n+i]) * float64(iv[k*n+j])
				}
				out[i*n+j] = float32(s)
				out[j*n+i] = float32(s)
			}
		}
	})
	return out
}

// cholUpper overwrites a with the upper triangular R of a = RᵀR.
func cholUpper(a []float32, n int) bool {
	for j := 0; j < n; j++ {
		var s float64
		for k := 0; k < j; k++ {
			s += float64(a[k*n+j]) * float64(a[k*n+j])
		}
		v := float64(a[j*n+j]) - s
		if v <= 0 {
			return false
		}
		d := math.Sqrt(v)
		a[j*n+j] = float32(d)
		for i := j + 1; i < n; i++ {
			var t float64
			for k := 0; k < j; k++ {
				t += float64(a[k*n+j]) * float64(a[k*n+i])
			}
			a[j*n+i] = float32((float64(a[j*n+i]) - t) / d)
		}
		for i := j + 1; i < n; i++ {
			a[i*n+j] = 0
		}
	}
	return true
}
