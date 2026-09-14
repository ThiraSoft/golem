//go:build amd64

package nn

//go:noescape
func matF16x4x3AVX2(w *uint16, wStride int, x *float32, xStride, n int, out *float32)

//go:noescape
func matF32x4x3AVX2(w *float32, wStride int, x *float32, xStride, n int, out *float32)

func tilesAvailable() bool { return avx2 }

// matF16Tile computes a four-row, three-column tile of an fp16 product.
func matF16Tile(w []uint16, wStride int, x []float32, xStride, n int, out *[12]float32) {
	matF16x4x3AVX2(&w[0], wStride, &x[0], xStride, n, &out[0])
}

// matF32Tile is the same tile with float32 rows.
func matF32Tile(w []float32, wStride int, x []float32, xStride, n int, out *[12]float32) {
	matF32x4x3AVX2(&w[0], wStride, &x[0], xStride, n, &out[0])
}
