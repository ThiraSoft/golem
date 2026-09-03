//go:build amd64

package nn

//go:noescape
func dotQ3_KAVX2(w *byte, q *int8, bsums *int16, scales *float32, n int) float32

func dotQ3_K(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	if avx2 {
		return dotQ3_KAVX2(&w[0], &q[0], &bsums[0], &scales[0], n)
	}
	return dotQ3_KGo(w, q, bsums, scales, n)
}
