//go:build amd64

package nn

//go:noescape
func dotQ6_KAVX2(w *byte, q *int8, bsums *int16, scales *float32, n int) float32

//go:noescape
func dotQ6_Kx2AVX2(w *byte, q0, q1 *int8, bsums0, bsums1 *int16, scales0, scales1 *float32, n int, out *float32)

// dotQ6_Kx2 is the same product for two activations, unpacking the row once.
// It reports whether it ran: without AVX2 there is nothing but the one-column
// path, and the caller loops.
func dotQ6_Kx2(w []byte, q0, q1 []int8, bsums0, bsums1 []int16, scales0, scales1 []float32, n int, out *[2]float32) bool {
	if !avx2 {
		return false
	}
	dotQ6_Kx2AVX2(&w[0], &q0[0], &q1[0], &bsums0[0], &bsums1[0], &scales0[0], &scales1[0], n, &out[0])
	return true
}

func dotQ6_K(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	if avx2 {
		return dotQ6_KAVX2(&w[0], &q[0], &bsums[0], &scales[0], n)
	}
	return dotQ6_KGo(w, q, bsums, scales, n)
}
