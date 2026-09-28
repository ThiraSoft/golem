package nn

// A blocked product for an fp16 matrix against a wide batch of float32
// columns, which is where an encoder that sees a few hundred to a few
// thousand positions at once spends its time on the processor.
//
// MatMulF16 widens a row of weights for every three columns it meets, and the
// widening competes with the multiply-adds for the same ports: it ran at
// about half of what the machine does. Here the matrix is laid out once in
// panels of sixteen rows with the shared dimension outermost, so that sixteen
// weights of one step are two loads, and a tile of sixteen outputs by six
// columns keeps its twelve accumulators in registers for a whole block of the
// shared dimension: each weight widened once meets six columns and each
// column's value broadcast once meets sixteen weights.
//
// The work is cut into tasks of thirty-two rows by a block of columns. In a
// pass of a few blocks the blocks of one set of rows are next to each other
// in the order the workers take them, so that the rows' weights come back
// from the last-level cache and not from memory; in a wider one it is the
// other way round, a block of columns meeting every set of rows in turn,
// because then it is the columns that would not stay. Each task walks the shared dimension in blocks small enough that the six
// columns of the tile stay in the first-level cache while both panels meet
// them.
//
// Every entry is the same float whichever task it falls in: its sums go in
// the order of the shared dimension, a block at a time, whatever the width of
// the pass.

import "unsafe"

// PackedF16 is an fp16 matrix laid out for GemmF16.
type PackedF16 struct {
	Rows, Cols int
	data       []uint16 // panels of sixteen rows, [Rows/16][Cols][16]
}

// GemmF16Available says whether this machine has the kernel. Without it
// PackF16 declines.
func GemmF16Available() bool { return avx2 }

// PackF16 lays m out for GemmF16. It reports false when m is not fp16, when
// its rows are not a whole number of sixteens or its columns of eights, or
// when the machine has no kernel.
func PackF16(m Matrix) (PackedF16, bool) {
	if m.Quant != F16 || m.Rows%16 != 0 || m.Cols%8 != 0 || !avx2 {
		return PackedF16{}, false
	}
	src := unsafe.Slice((*uint16)(unsafe.Pointer(&m.Data[0])), m.Rows*m.Cols)
	p := PackedF16{Rows: m.Rows, Cols: m.Cols, data: make([]uint16, m.Rows*m.Cols)}
	InParallel(m.Rows/16, m.Rows*m.Cols, func(first, last int) {
		for panel := first; panel < last; panel++ {
			dst := p.data[panel*16*m.Cols : (panel+1)*16*m.Cols]
			for r := 0; r < 16; r++ {
				row := src[(panel*16+r)*m.Cols : (panel*16+r+1)*m.Cols]
				for k, h := range row {
					dst[k*16+r] = h
				}
			}
		}
	})
	return p, true
}

// gemmKC is the block of the shared dimension a tile walks: six columns of
// it are six kilobytes, and the two panels that meet them sixteen.
const gemmKC = 256

// gemmNB is how many columns one task takes.
const gemmNB = 48

// GemmF16 computes out[c*Rows+r] = Σ_k W[r][k]·x[c*Cols+k] (+ bias[r]) for
// c < cols. x holds the columns back to back, and so does out. With add the
// product is added to what out holds instead, which is a residual stream
// taking a block's output without a pass of its own.
func (p PackedF16) GemmF16(x []float32, cols int, out []float32, bias []float32, add bool) {
	if cols == 0 {
		return
	}
	if len(x) < cols*p.Cols || len(out) < cols*p.Rows {
		panic("nn: GemmF16 was handed fewer floats than its columns hold")
	}
	groups := p.Rows / 32
	half := p.Rows%32 != 0 // a last group of sixteen
	if half {
		groups++
	}
	blocks := (cols + gemmNB - 1) / gemmNB
	InParallel(groups*blocks, p.Rows*p.Cols*cols*2, func(first, last int) {
		var edgeX [6 * gemmKC]float32
		var edgeC [6 * 16]float32
		for task := first; task < last; task++ {
			g, b := task/blocks, task%blocks
			if blocks > 8 {
				g, b = task%groups, task/groups
			}
			r0 := g * 32
			panels := min(2, (p.Rows-r0)/16)
			c0, c1 := b*gemmNB, min((b+1)*gemmNB, cols)
			// Each task clears what it adds into: a pass's outputs are
			// megabytes, and clearing them before the section was a
			// fifteenth of it on one core. With add the sums start from
			// what is there, so each block of the shared dimension is
			// added to it in turn.
			if !add {
				for c := c0; c < c1; c++ {
					clear(out[c*p.Rows+r0 : c*p.Rows+r0+panels*16])
				}
			}
			for k0 := 0; k0 < p.Cols; k0 += gemmKC {
				kc := min(gemmKC, p.Cols-k0)
				for c := c0; c < c1; c += 6 {
					n := min(6, c1-c)
					xs, stride := unsafe.Pointer(&x[c*p.Cols+k0]), p.Cols*4
					if n < 6 {
						// The last columns are copied into a tile of six,
						// zeros after them, and their sums added back.
						clear(edgeX[:])
						for j := 0; j < n; j++ {
							copy(edgeX[j*kc:(j+1)*kc], x[(c+j)*p.Cols+k0:(c+j)*p.Cols+k0+kc])
						}
						xs, stride = unsafe.Pointer(&edgeX[0]), kc*4
					}
					for q := 0; q < panels; q++ {
						w := &p.data[(r0/16+q)*16*p.Cols+k0*16]
						if n == 6 {
							gemmF16x16x6AVX2(w, (*float32)(xs), stride, kc, &out[c*p.Rows+r0+q*16], p.Rows*4)
							continue
						}
						clear(edgeC[:])
						gemmF16x16x6AVX2(w, (*float32)(xs), stride, kc, &edgeC[0], 64)
						for j := 0; j < n; j++ {
							dst := out[(c+j)*p.Rows+r0+q*16:]
							for r := 0; r < 16; r++ {
								dst[r] += edgeC[j*16+r]
							}
						}
					}
				}
			}
			if bias != nil {
				for c := c0; c < c1; c++ {
					dst := out[c*p.Rows+r0 : c*p.Rows+r0+panels*16]
					for r := range dst {
						dst[r] += bias[r0+r]
					}
				}
			}
		}
	})
}
