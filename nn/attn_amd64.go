//go:build amd64

package nn

//go:noescape
func scoresF32AVX2(q *float32, k *float32, hd, n int, out *float32)

//go:noescape
func mixF32AVX2(dst *float32, v *float32, w *float32, hd, n int)

// Scores writes the dot product of q with each of the n heads laid end to end
// in k. It is the inner half of one head's attention.
func Scores(q, k []float32, hd, n int, out []float32) {
	if avx2 {
		scoresF32AVX2(&q[0], &k[0], hd, n, &out[0])
		return
	}
	for j := 0; j < n; j++ {
		out[j] = DotF32(q, k[j*hd:(j+1)*hd])
	}
}

// Mix adds the weighted sum of the n heads laid end to end in v into dst,
// which is the other half.
func Mix(dst, v, w []float32, hd, n int) {
	if avx2 {
		mixF32AVX2(&dst[0], &v[0], &w[0], hd, n)
		return
	}
	for j := 0; j < n; j++ {
		Axpy(dst, v[j*hd:(j+1)*hd], w[j])
	}
}

//go:noescape
func softmaxAVX2(x *float32, n int)

// SoftmaxGGML normalizes x into probabilities the way ggml does: the maximum
// subtracted so the exponential cannot overflow, and that exponential its own
// polynomial rather than the library's.
//
// It is the softmax of the vision tower, where the exponential is called two
// hundred million times for one picture. SoftmaxInPlace stays as it is for the
// language models, whose fixtures were recorded against it.
func SoftmaxGGML(x []float32) {
	if avx2 {
		softmaxAVX2(&x[0], len(x))
		return
	}
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

//go:noescape
func scores4AVX2(q *float32, k *float32, hd, n int, out *float32, outStride int)

//go:noescape
func mix4AVX2(dst *float32, dstStride int, v *float32, w *float32, wStride, hd, n int)

// Scores4 writes four rows of scores at once: query i of the four against
// every one of the n heads in k, into out[i*outStride:].
//
// It reports whether it could. The kernel wants a head that is a whole number
// of eights; anything else is the caller's to do a query at a time.
func Scores4(q, k []float32, hd, n int, out []float32, outStride int) bool {
	if !avx2 || hd%8 != 0 {
		return false
	}
	scores4AVX2(&q[0], &k[0], hd, n, &out[0], outStride)
	return true
}

// Mix4 adds four weighted sums of the same n heads into four destinations,
// dstStride apart, with the weights of query i at w[i*wStride:].
func Mix4(dst []float32, dstStride int, v, w []float32, wStride, hd, n int) bool {
	if !avx2 || hd%16 != 0 {
		return false
	}
	mix4AVX2(&dst[0], dstStride, &v[0], &w[0], wStride, hd, n)
	return true
}
