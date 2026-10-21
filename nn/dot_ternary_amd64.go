//go:build amd64

package nn

//go:noescape
func dotPQ2_0AVX2(w *byte, q *int8, sums *int32, scales *float32, blocks int) float32

//go:noescape
func dotPTQ1_0AVX2(w *byte, q *int8, sums *int32, scales *float32, blocks int) float32

func dotPQ2_0(w []byte, a *ternaryColumn, n int) float32 {
	if avx2 {
		return dotPQ2_0AVX2(&w[0], &a.q[0], &a.sums[0], &a.scales[0], n/TernaryBlock)
	}
	return dotPQ2_0Go(w, a, n)
}

func dotPTQ1_0(w []byte, a *ternaryColumn, n int) float32 {
	if avx2 {
		return dotPTQ1_0AVX2(&w[0], &a.q[0], &a.sums[0], &a.scales[0], n/TernaryBlock)
	}
	return dotPTQ1_0Go(w, a, n)
}
