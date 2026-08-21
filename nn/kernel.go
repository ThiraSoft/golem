package nn

// Matrix-vector product reading the weights directly in bfloat16.
//
// Converting the whole model to float32 once and for all would cost 1.26 GB of
// memory and, above all, would double the bandwidth to re-read on every frame —
// and bandwidth is what caps generation. The kernel therefore works on the
// original bytes and converts on the fly: a bfloat16 is a float32 stripped of
// its low mantissa, so the shift is enough.

import (
	"math"
	"unsafe"
)

// parallelThreshold is the amount of work below which spreading over several
// cores costs more than computing.
//
// The engine alternates between two very different regimes: the large matrices
// of the flow_lm, where parallelism is indispensable, and the small ones of the
// flow net and the audio decoder — hundreds per frame, each too short to amortize
// launching eight goroutines. Without this threshold, synchronization cost more
// than the computation.
const parallelThreshold = 1 << 19

// MatVecBF16 computes y = W*x, with W stored row-major in little-endian
// bfloat16.
func MatVecBF16(w []byte, x []float32, outputs, inputs int, y []float32) {
	InParallel(outputs, outputs*inputs, func(start, end int) {
		MatVecBF16Rows(w, x, inputs, y, start, end)
	})
}

// MatVecBF16Rows computes rows [start, end) on the caller's thread, for a
// caller that is already inside a section and has something to do with the rows
// it just produced before the barrier.
func MatVecBF16Rows(w []byte, x []float32, inputs int, y []float32, start, end int) {
	weights := unsafe.Slice((*uint16)(unsafe.Pointer(&w[0])), len(w)/2)
	for o := start; o < end; o++ {
		y[o] = dotBF16(weights[o*inputs:(o+1)*inputs], x)
	}
}

// dotBF16 is one row of bfloat16 weights against one activation.
func dotBF16(row []uint16, x []float32) float32 {
	if len(row) == 0 {
		return 0
	}
	if avx2 {
		return dotBF16AVX2(&row[0], &x[0], len(row))
	}
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < len(row); i += 4 {
		s0 += math.Float32frombits(uint32(row[i])<<16) * x[i]
		s1 += math.Float32frombits(uint32(row[i+1])<<16) * x[i+1]
		s2 += math.Float32frombits(uint32(row[i+2])<<16) * x[i+2]
		s3 += math.Float32frombits(uint32(row[i+3])<<16) * x[i+3]
	}
	for ; i < len(row); i++ {
		s0 += math.Float32frombits(uint32(row[i])<<16) * x[i]
	}
	return s0 + s1 + s2 + s3
}

// MatMatBF16 computes Y = W*X for `batch` vectors at a time, with X and Y
// stored vector by vector.
//
// It is the same operation as MatVecBF16 repeated, with one detail that changes
// everything: each row of weights is read only once for the whole batch. In
// generation the limiting factor is re-reading the weights — a transformer that
// processes sixteen positions in a row while reading its matrices sixteen times
// spends its time waiting on memory for nothing.
func MatMatBF16(w []byte, x []float32, outputs, inputs, batch int, y []float32) {
	InParallel(outputs, outputs*inputs*batch, func(start, end int) {
		MatMatBF16Rows(w, x, outputs, inputs, batch, y, start, end)
	})
}

// panelBytes is how much of the second-level cache a panel of activations may
// take. A core here has 256 KB of it and the rows of weights streaming past
// want their share, so a panel is given half.
const panelBytes = 128 << 10

// panelFloor is the narrowest panel worth having. A wide input makes for a
// narrow panel, and a narrow panel is a matrix read again for every handful of
// columns: the activations of a wider one spill out of the second-level cache,
// but the third holds them and the bus stops carrying the weights over and
// over.
const panelFloor = 8

// panelOf is how many vectors of the batch are taken together: enough that the
// weights are read from memory a handful of times for a whole batch, few
// enough that the panel answers from cache.
func panelOf(inputs int) int {
	n := panelBytes / (inputs * 4)
	if n < panelFloor {
		n = panelFloor
	}
	// A whole number of fours: what does not fill a group of four columns
	// falls to the one-column kernel, which is four times the work per
	// weight read, and two stray columns in every panel are not two columns
	// in a hundred of the time.
	n &^= 3
	if n < 4 {
		return 4
	}
	return n
}

// MatMatBF16Rows is the same for rows [start, end) on the caller's thread.
func MatMatBF16Rows(w []byte, x []float32, outputs, inputs, batch int, y []float32, start, end int) {
	weights := unsafe.Slice((*uint16)(unsafe.Pointer(&w[0])), len(w)/2)

	// The row is converted once for the whole batch: without that, the
	// bfloat16 conversion would be paid as many times as there are
	// vectors, which is precisely what the batch is meant to amortize.
	// With AVX2 the question no longer arises: the raw row fits in the
	// first-level cache, re-reading it for each vector of the batch
	// costs nothing, and the conversion is folded into the computation.
	//
	// Four columns at a time where there are four: widening a row of weights
	// costs the same whether one column or four are waiting for it, and the
	// one-column kernel spends two instructions in three feeding the
	// multiply-accumulate rather than doing it.
	//
	// And the batch is walked in panels rather than whole. A large batch is
	// wider than any cache: taking all of it for one row of weights, and then
	// all of it again for the next, streams the activations past every row —
	// megabytes per row, from memory, for a row of weights that is kilobytes.
	// A panel narrow enough to sit in the second-level cache is read once from
	// memory and then answers every row out of it.
	if avx2 {
		var four [4]float32
		var eight [8]float32
		panel := panelOf(inputs)
		// Two rows at a time when the row is a whole number of eights, which
		// is what makes an activation load feed two multiply-accumulates
		// instead of one.
		pairs := inputs%8 == 0
		for at := 0; at < batch; at += panel {
			to := min(at+panel, batch)
			o := start
			if pairs {
				for ; o+1 < end; o += 2 {
					raw := weights[o*inputs:]
					l := at
					for ; l+3 < to; l += 4 {
						dotBF16x2x4AVX2(&raw[0], inputs, &x[l*inputs], inputs, inputs, &eight[0])
						for c := 0; c < 4; c++ {
							y[(l+c)*outputs+o] = eight[c]
							y[(l+c)*outputs+o+1] = eight[4+c]
						}
					}
					for ; l < to; l++ {
						y[l*outputs+o] = dotBF16AVX2(&raw[0], &x[l*inputs], inputs)
						y[l*outputs+o+1] = dotBF16AVX2(&weights[(o+1)*inputs], &x[l*inputs], inputs)
					}
				}
			}
			for ; o < end; o++ {
				raw := weights[o*inputs : (o+1)*inputs]
				l := at
				for ; l+3 < to; l += 4 {
					dotBF16x4AVX2(&raw[0], &x[l*inputs], inputs, inputs, &four[0])
					y[l*outputs+o] = four[0]
					y[(l+1)*outputs+o] = four[1]
					y[(l+2)*outputs+o] = four[2]
					y[(l+3)*outputs+o] = four[3]
				}
				for ; l < to; l++ {
					y[l*outputs+o] = dotBF16AVX2(&raw[0], &x[l*inputs], inputs)
				}
			}
		}
		return
	}
	row := make([]float32, inputs)
	for o := start; o < end; o++ {
		raw := weights[o*inputs : (o+1)*inputs]
		for i, p := range raw {
			row[i] = math.Float32frombits(uint32(p) << 16)
		}
		for l := 0; l < batch; l++ {
			v := x[l*inputs : (l+1)*inputs]
			var s0, s1, s2, s3 float32
			i := 0
			for ; i+3 < inputs; i += 4 {
				s0 += row[i] * v[i]
				s1 += row[i+1] * v[i+1]
				s2 += row[i+2] * v[i+2]
				s3 += row[i+3] * v[i+3]
			}
			for ; i < inputs; i++ {
				s0 += row[i] * v[i]
			}
			y[l*outputs+o] = s0 + s1 + s2 + s3
		}
	}
}

// Axpy computes dst += a*src over the first n elements common to both.
//
// This is the inner loop of the convolutions; it deserves its own kernel just
// as much as the matrix-vector product does.
func Axpy(dst, src []float32, a float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	if avx2 {
		axpyAVX2(&dst[0], &src[0], n, a)
		return
	}
	for i := 0; i < n; i++ {
		dst[i] += a * src[i]
	}
}

// AxpyFull is Axpy without the guard: dst must be at least as long as src. The
// convolutions call the loop hundreds of times per frame on channels a few
// positions long; at that scale, computing the length and testing for overrun
// weigh as much as the computation itself.
func AxpyFull(dst, src []float32, a float32) {
	if len(src) == 0 {
		return
	}
	if avx2 {
		axpyAVX2(&dst[0], &src[0], len(src), a)
		return
	}
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] += a * v
	}
}

// Interleaved says whether the interleaved kernel exists for a row of this
// width, and therefore whether the caller should lay its activation out for
// it. Everything narrower, or a machine without AVX2, takes the plain path.
func Interleaved(inputs int) bool { return avx2 && inputs%16 == 0 }

// InterleaveOrder is the permutation the interleaved kernel reads a group of
// sixteen in. A caller writing an activation for it writes group[i] where the
// plain layout would have written group[InterleaveOrder[i]].
var InterleaveOrder = [16]int{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15}

// MatMatBF16IlvRows is MatMatBF16Rows for an activation already laid out in
// that order. It is what the vision tower uses, and it is a fifth faster:
// widening the weights by an interleave instead of a shift takes two
// instructions per eight products off the ports the arithmetic runs on.
func MatMatBF16IlvRows(w []byte, x []float32, outputs, inputs, batch int, y []float32, start, end int) {
	weights := unsafe.Slice((*uint16)(unsafe.Pointer(&w[0])), len(w)/2)
	var eight [8]float32
	panel := panelOf(inputs)
	// The columns outside and the rows inside, which is the order that keeps
	// both operands where they are wanted: four columns are twelve kilobytes
	// and stay in the first-level cache while every row of this core's share
	// sweeps past them, and that share of the weights is small enough to stay
	// in the second.
	for at := 0; at < batch; at += panel {
		to := min(at+panel, batch)
		l := at
		for ; l+3 < to; l += 4 {
			o := start
			for ; o+1 < end; o += 2 {
				dotBF16x2x4IlvAVX2(&weights[o*inputs], inputs, &x[l*inputs], inputs, inputs, &eight[0])
				for c := 0; c < 4; c++ {
					y[(l+c)*outputs+o] = eight[c]
					y[(l+c)*outputs+o+1] = eight[4+c]
				}
			}
			for ; o < end; o++ {
				dotBF16x2x4IlvAVX2(&weights[o*inputs], 0, &x[l*inputs], inputs, inputs, &eight[0])
				for c := 0; c < 4; c++ {
					y[(l+c)*outputs+o] = eight[c]
				}
			}
		}
		// The tail of a panel that is not a whole number of fours.
		for ; l < to; l++ {
			o := start
			for ; o+1 < end; o += 2 {
				dotBF16x2x4IlvAVX2(&weights[o*inputs], inputs, &x[l*inputs], 0, inputs, &eight[0])
				y[l*outputs+o] = eight[0]
				y[l*outputs+o+1] = eight[4]
			}
			for ; o < end; o++ {
				dotBF16x2x4IlvAVX2(&weights[o*inputs], 0, &x[l*inputs], 0, inputs, &eight[0])
				y[l*outputs+o] = eight[0]
			}
		}
	}
}

// unsafeWords views a byte slice as the bfloat16 words it holds, for the tests
// that have to write weights rather than read them.
func unsafeWords(b []byte) []uint16 {
	return unsafe.Slice((*uint16)(unsafe.Pointer(&b[0])), len(b)/2)
}
