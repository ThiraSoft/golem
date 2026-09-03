//go:build !amd64 && !arm64

package nn

func dotQ3_K(w []byte, q []int8, bsums []int16, scales []float32, n int) float32 {
	return dotQ3_KGo(w, q, bsums, scales, n)
}
