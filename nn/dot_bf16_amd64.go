//go:build amd64

package nn

// hasBF16x4 says the four-column kernel exists here, which on this
// architecture is the same question as whether the machine has AVX2.
var hasBF16x4 = avx2

// fastDotBF16 is the AVX2 row product, when the machine has AVX2.
func fastDotBF16(row *uint16, x *float32, n int) (float32, bool) {
	if !avx2 {
		return 0, false
	}
	return dotBF16AVX2(row, x, n), true
}
