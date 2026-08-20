//go:build !amd64

package nn

func dotQ6_K(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	return dotQ6_KGo(w, q, bsums, scales, n)
}
