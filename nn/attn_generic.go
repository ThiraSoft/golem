//go:build !amd64

package nn

// Scores writes the dot product of q with each of the n heads laid end to end
// in k. It is the inner half of one head's attention.
func Scores(q, k []float32, hd, n int, out []float32) {
	for j := 0; j < n; j++ {
		out[j] = DotF32(q, k[j*hd:(j+1)*hd])
	}
}

// Mix adds the weighted sum of the n heads laid end to end in v into dst.
func Mix(dst, v, w []float32, hd, n int) {
	for j := 0; j < n; j++ {
		Axpy(dst, v[j*hd:(j+1)*hd], w[j])
	}
}

// SoftmaxGGML normalizes x into probabilities the way ggml does, with its own
// exponential rather than the library's.
func SoftmaxGGML(x []float32) {
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		e := expf(v - max)
		x[i] = e
		sum += e
	}
	for i := range x {
		x[i] /= sum
	}
}

// Scores4 and Mix4 are the four-at-a-time forms, which exist only where there
// is a kernel for them.
func Scores4(q, k []float32, hd, n int, out []float32, outStride int) bool { return false }

func Mix4(dst []float32, dstStride int, v, w []float32, wStride, hd, n int) bool { return false }

// Clamp holds every value of x inside [lo, hi].
func Clamp(x []float32, lo, hi float32) {
	for i, v := range x {
		if v < lo {
			x[i] = lo
		} else if v > hi {
			x[i] = hi
		}
	}
}
