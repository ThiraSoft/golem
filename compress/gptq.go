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
// That factoring machinery — Comp, NewComp, the Cholesky routines — is gone.
// It is a measured, closed question and not an open one: on activations
// correlated the way a test could make them it halved the output error, but on
// Qwen3-0.6B it moved the perplexity by 0.15 of a point out of a nine-point
// gap, because incoherence processing already pushes the Hessian towards a
// multiple of the identity — against a diagonal Hessian the columns no longer
// overlap, so there is no error left to pass on. `cmd/golemquant`'s -gptq and
// -damp flags read this same conclusion and were removed with it. What Acc
// keeps is the part something else still reads: the windowed Hessian
// AddRows accumulates is also what Energy scores a candidate salience
// against, which is a live question and not a closed one.

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
// the weights were. Only the upper triangle of each window is written. The
// work is split by window rather than by row because the windows are what the
// accumulators are, and two rows of the same window would have to take turns.
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

// Energy is what one row's error costs the product: dᵀHd summed over the
// windows, where d is the difference between the row and what was stored, in
// the basis the activations were measured in.
//
// This is the number the salience is chosen against. Weight error is not: the
// whole point of scaling a column is to move error from where the activations
// are large to where they are small, and a metric that weighs every column
// alike cannot see that happening. The windows drop the correlation between
// columns more than a few hundred apart, which is the same approximation the
// (now retired) compensation pass made and for the same reason.
func (a *Acc) Energy(d []float32) float64 {
	var total float64
	for k := range a.blocks {
		m := a.sizes[k]
		h := a.blocks[k]
		v := d[k*a.window : k*a.window+m]
		var s float64
		for i := 0; i < m; i++ {
			vi := float64(v[i])
			if vi == 0 {
				continue
			}
			hi := h[i*m : i*m+m]
			// The upper triangle is what was accumulated; the diagonal counts
			// once and everything above it twice.
			s += vi * float64(hi[i]) * vi
			var off float64
			for j := i + 1; j < m; j++ {
				off += float64(hi[j]) * float64(v[j])
			}
			s += 2 * vi * off
		}
		total += s
	}
	return total
}
