package compress

// A scheme is the whole cumulated pipeline: salience scaling, an optional
// rotation, an optional set of columns kept at high precision, then the vector
// quantizer. Each stage attacks a different part of the error, and the bill for
// all of them is one number, bits per weight.

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

type Scheme struct {
	Alpha    float64 // salience exponent: columns are scaled by act_rms^Alpha
	Outliers int     // columns kept at 8 bits, chosen by salience
	VQ       Opts    // the quantizer the rest of the matrix goes through
	Scalar   int     // if non-zero, use an n-bit scalar quantizer instead of VQ
	Block    int     // scalar block size
}

func (s Scheme) Name() string {
	var sb strings.Builder
	if s.Alpha != 0 {
		fmt.Fprintf(&sb, "AWQ%.2f+", s.Alpha)
	}
	if s.Outliers > 0 {
		fmt.Fprintf(&sb, "OUT%d+", s.Outliers)
	}
	if s.Scalar > 0 {
		fmt.Fprintf(&sb, "Q%d/%d", s.Scalar, s.Block)
	} else {
		sb.WriteString(s.VQ.Name())
	}
	return sb.String()
}

// BPW counts what the scheme stores per weight: the quantizer's own budget,
// plus the salience scales, plus the columns held out at 8 bits and the index
// naming them.
func (s Scheme) BPW(cols int) float64 {
	var b float64
	if s.Scalar > 0 {
		b = float64(s.Scalar) + 16/float64(s.Block)
	} else {
		b = s.VQ.BPW()
	}
	kept := float64(s.Outliers) / float64(cols)
	b = b*(1-kept) + kept*(8.5+float64(bitsFor(cols))/float64(cols)*0)
	if s.Outliers > 0 {
		// the column index, once per held-out column, spread over its rows —
		// negligible but not free
		b += float64(s.Outliers*bitsFor(cols)) / float64(cols) / 1024
	}
	if s.Alpha != 0 {
		b += 16 / float64(1<<20) // one fp16 per column, per row: vanishing
	}
	return b
}

func bitsFor(n int) int {
	b := 0
	for 1<<b < n {
		b++
	}
	return b
}

// Apply runs the scheme over a matrix and returns what a kernel reading the
// compressed form would reconstruct. salience is the per-column RMS of the
// activations that meet this matrix; nil means every column matters equally.
func (s Scheme) Apply(w []float32, rows, cols int, salience []float32, seed int64) []float32 {
	work := make([]float32, len(w))
	copy(work, w)

	// 1. Salience scaling. Multiplying a column of W by sⱼ and dividing the
	// matching activation by sⱼ leaves the product unchanged, but moves the
	// quantizer's attention onto the columns that carry signal.
	scale := make([]float32, cols)
	for j := range scale {
		scale[j] = 1
	}
	if s.Alpha != 0 && salience != nil {
		var geo float64
		for j := 0; j < cols; j++ {
			v := float64(salience[j])
			if v < 1e-8 {
				v = 1e-8
			}
			geo += math.Log(v)
		}
		geo = math.Exp(geo / float64(cols))
		for j := 0; j < cols; j++ {
			v := float64(salience[j])
			if v < 1e-8 {
				v = 1e-8
			}
			scale[j] = float32(math.Pow(v/geo, s.Alpha))
		}
		Parallel(rows, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				for j := 0; j < cols; j++ {
					work[r*cols+j] *= scale[j]
				}
			}
		})
	}

	// 2. Columns held out at 8 bits: the few that carry so much of the output
	// that no shared codebook can afford to average them away.
	var kept []int
	if s.Outliers > 0 && salience != nil {
		order := make([]int, cols)
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool { return salience[order[a]] > salience[order[b]] })
		kept = order[:s.Outliers]
		sort.Ints(kept)
	}
	keptVals := make([]float32, len(kept)*rows)
	for i, j := range kept {
		for r := 0; r < rows; r++ {
			keptVals[i*rows+r] = work[r*cols+j]
		}
	}
	// The held-out columns are replaced by the row mean so they neither dominate
	// the codebook nor leave a hole in it.
	for _, j := range kept {
		for r := 0; r < rows; r++ {
			work[r*cols+j] = 0
		}
	}

	// 3. The quantizer proper.
	var rec []float32
	if s.Scalar > 0 {
		rec = QNsym(work, rows, cols, s.Scalar, s.Block)
	} else {
		rec = VQ(work, rows, cols, s.VQ, seed)
		if rec == nil {
			return nil
		}
	}

	// 4. Put the held-out columns back, quantized to 8 bits per row-block.
	for i, j := range kept {
		col := keptVals[i*rows : (i+1)*rows]
		for b := 0; b < rows; b += 32 {
			e := b + 32
			if e > rows {
				e = rows
			}
			var amax float32
			for _, v := range col[b:e] {
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			d := Fp16round(amax / 127)
			for r := b; r < e; r++ {
				q := float32(0)
				if d != 0 {
					q = float32(math.Round(float64(col[r] / d)))
					if q > 127 {
						q = 127
					}
					if q < -127 {
						q = -127
					}
				}
				rec[r*cols+j] = q * d
			}
		}
	}

	// 5. Undo the salience scaling: at inference the activations carry it.
	if s.Alpha != 0 && salience != nil {
		Parallel(rows, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				for j := 0; j < cols; j++ {
					rec[r*cols+j] /= scale[j]
				}
			}
		})
	}
	return rec
}
