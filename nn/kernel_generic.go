//go:build !amd64

package nn

const avx2 = false

func dotBF16AVX2(row *uint16, x *float32, n int) float32 { panic("unavailable") }

func axpyAVX2(dst, src *float32, n int, a float32) { panic("unavailable") }

func dotF32AVX2(a, b *float32, n int) float32 { panic("unavailable") }

func dotBF16x4AVX2(row *uint16, x *float32, stride, n int, out *float32) { panic("unavailable") }

func dotBF16x2x4AVX2(row *uint16, rowStride int, x *float32, colStride, n int, out *float32) {
	panic("unavailable")
}

func dotBF16x2x4IlvAVX2(row *uint16, rowStride int, x *float32, colStride, n int, out *float32) {
	panic("unavailable")
}
