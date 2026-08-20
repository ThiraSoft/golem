//go:build amd64

package nn

//go:noescape
func dotQ6_KAVX2(w *byte, q *int8, bsums *int16, scales *float32, n int) float32

func dotQ6_K(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	if avx2 {
		return dotQ6_KAVX2(&w[0], &q[0], &bsums[0], &scales[0], n)
	}
	return dotQ6_KGo(w, q, bsums, scales, n)
}
