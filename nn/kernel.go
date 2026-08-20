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
		matVecBF16Rows(w, x, inputs, y, start, end)
	})
}

// matVecBF16Rows computes rows [start, end) on the caller's thread.
func matVecBF16Rows(w []byte, x []float32, inputs int, y []float32, start, end int) {
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
	weights := unsafe.Slice((*uint16)(unsafe.Pointer(&w[0])), len(w)/2)

	InParallel(outputs, outputs*inputs*batch, func(start, end int) {
		{
			// The row is converted once for the whole batch: without that, the
			// bfloat16 conversion would be paid as many times as there are
			// vectors, which is precisely what the batch is meant to amortize.
			// With AVX2 the question no longer arises: the raw row fits in the
			// first-level cache, re-reading it for each vector of the batch
			// costs nothing, and the conversion is folded into the computation.
			if avx2 {
				for o := start; o < end; o++ {
					raw := weights[o*inputs : (o+1)*inputs]
					for l := 0; l < batch; l++ {
						y[l*outputs+o] = dotBF16AVX2(&raw[0], &x[l*inputs], inputs)
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
	})
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
