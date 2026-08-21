//go:build !amd64

package nn

// Without AVX2 there is one path, and the caller takes the columns one at a
// time.
func dotBF16x4(row []uint16, x []float32, stride, n int, out *[4]float32) bool {
	return false
}
