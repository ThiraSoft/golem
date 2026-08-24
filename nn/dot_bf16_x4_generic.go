//go:build !amd64 && !arm64

package nn

// hasBF16x4 says there is no four-column kernel here.
const hasBF16x4 = false

// Without a four-column kernel there is one path, and the caller takes the
// columns one at a time.
func dotBF16x4(row []uint16, x []float32, stride, n int, out *[4]float32) bool {
	return false
}
