//go:build !amd64

package nn

// Without AVX2 there is one path, and the caller takes the columns one at a
// time.
func dotQ6_Kx2(w []byte, q0, q1 []int8, bsums0, bsums1 []int16, scales0, scales1 []float32, n int, out *[2]float32) bool {
	return false
}
