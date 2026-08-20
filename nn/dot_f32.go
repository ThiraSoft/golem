package nn

// The plain float32 dot product.
//
// Attention scores are the one product in the engine whose operands are both
// activations: a query against a key already in the cache. There is no
// quantization to hide behind, and at one position per token the loop is short
// — short enough that what limits it is the dependency between the additions,
// not the arithmetic. Four accumulators break that chain; the kernel widens
// each of them to eight lanes.

// DotF32 returns the dot product of the first n elements common to a and b.
func DotF32(a, b []float32) float32 {
	n := min(len(a), len(b))
	if n == 0 {
		return 0
	}
	if avx2 {
		return dotF32AVX2(&a[0], &b[0], n)
	}
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}
