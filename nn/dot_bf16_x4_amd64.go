//go:build amd64

package nn

//go:noescape
func dotBF16x4AVX2(row *uint16, x *float32, stride, n int, out *float32)

// dotBF16x4 computes four dot products of one weight row against four
// activation columns, and reports whether it did: without AVX2 the caller
// takes the columns one at a time.
func dotBF16x4(row []uint16, x []float32, stride, n int, out *[4]float32) bool {
	if !avx2 {
		return false
	}
	dotBF16x4AVX2(&row[0], &x[0], stride, n, &out[0])
	return true
}

//go:noescape
func dotBF16x2x4AVX2(row *uint16, rowStride int, x *float32, colStride, n int, out *float32)
